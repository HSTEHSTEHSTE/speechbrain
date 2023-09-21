#!/usr/bin/env/python3
"""
Recipe for "direct" (speech -> semantics) SLU with ASR-based transfer learning.

We encode input waveforms into features using a model trained on LibriSpeech,
then feed the features into a seq2seq model to map them to semantics.

(Adapted from the LibriSpeech seq2seq ASR recipe written by Ju-Chieh Chou, Mirco Ravanelli, Abdel Heba, and Peter Plantinga.)

Run using:
> python train.py hparams/train.yaml

Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
"""

import sys
import torch
import gc
import speechbrain as sb
import logging
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import undo_padding

logger = logging.getLogger(__name__)

# Define training procedure


class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_with_bos, token_with_bos_lens = batch.tokens_bos
        # wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "augmentation"):
                feats = self.modules.augmentation(feats)

        x = self.modules.enc(feats.detach())
        e_in = self.modules.emb(tokens_with_bos)
        h, _ = self.modules.dec(e_in)
        # Joint network
        # add labelseq_dim to the encoder tensor: [B,T,H_enc] => [B,T,1,H_enc]
        # add timeseq_dim to the decoder tensor: [B,U,H_dec] => [B,1,U,H_dec]
        joint = self.modules.Tjoint(x.unsqueeze(2), h.unsqueeze(1))

        # Output layer for transducer log-probabilities
        logits_transducer = self.modules.transducer_lin(joint)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            return_CTC = False
            return_CE = False
            current_epoch = self.hparams.epoch_counter.current
            if (
                hasattr(self.hparams, "ctc_cost")
                and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                return_CTC = True
                # Output layer for ctc log-probabilities
                out_ctc = self.modules.enc_lin(x)
                p_ctc = self.hparams.log_softmax(out_ctc)
            if (
                hasattr(self.hparams, "ce_cost")
                and current_epoch <= self.hparams.number_of_ce_epochs
            ):
                return_CE = True
                # Output layer for ctc log-probabilities
                p_ce = self.modules.dec_lin(h)
                p_ce = self.hparams.log_softmax(p_ce)
            if return_CE and return_CTC:
                return p_ctc, p_ce, logits_transducer, wav_lens
            elif return_CTC:
                return p_ctc, logits_transducer, wav_lens
            elif return_CE:
                return p_ce, logits_transducer, wav_lens
            else:
                return logits_transducer, wav_lens

        elif stage == sb.Stage.VALID:
            best_hyps, scores, _, _ = self.hparams.beam_searcher(x)
            return logits_transducer, wav_lens, best_hyps
        else:
            (
                best_hyps,
                best_scores,
                nbest_hyps,
                nbest_scores,
            ) = self.hparams.beam_searcher(x)
            return logits_transducer, wav_lens, best_hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (Transducer+(CTC+NLL)) given predictions and targets."""

        ids = batch.id
        current_epoch = self.hparams.epoch_counter.current
        tokens, token_lens = batch.tokens
        tokens_eos, token_eos_lens = batch.tokens_eos

        if stage == sb.Stage.TRAIN:
            if len(predictions) == 4:
                p_ctc, p_ce, logits_transducer, wav_lens = predictions
                CTC_loss = self.hparams.ctc_cost(
                    p_ctc, tokens, wav_lens, token_lens
                )
                CE_loss = self.hparams.ce_cost(
                    p_ce, tokens_eos, length=token_eos_lens
                )
                loss_transducer = self.hparams.transducer_cost(
                    logits_transducer, tokens, wav_lens, token_lens
                )
                loss = (
                    self.hparams.ctc_weight * CTC_loss
                    + self.hparams.ce_weight * CE_loss
                    + (1 - (self.hparams.ctc_weight + self.hparams.ce_weight))
                    * loss_transducer
                )
            elif len(predictions) == 3:
                # one of the 2 heads (CTC or CE) is still computed
                # CTC alive
                if current_epoch <= self.hparams.number_of_ctc_epochs:
                    p_ctc, p_transducer, wav_lens = predictions
                    CTC_loss = self.hparams.ctc_cost(
                        p_ctc, tokens, wav_lens, token_lens
                    )
                    loss_transducer = self.hparams.transducer_cost(
                        logits_transducer, tokens, wav_lens, token_lens
                    )
                    loss = (
                        self.hparams.ctc_weight * CTC_loss
                        + (1 - self.hparams.ctc_weight) * loss_transducer
                    )
                # CE for decoder alive
                else:
                    p_ce, logits_transducer, wav_lens = predictions
                    CE_loss = self.hparams.ce_cost(
                        p_ce, tokens_eos, length=token_eos_lens
                    )
                    # Transducer loss use logits from RNN-T model.
                    loss_transducer = self.hparams.transducer_cost(
                        logits_transducer, tokens, wav_lens, token_lens
                    )
                    loss = (
                        self.hparams.ce_weight * CE_loss
                        + (1 - self.hparams.ctc_weight) * loss_transducer
                    )
            else:
                logits_transducer, wav_lens = predictions
                # Transducer loss use logits from RNN-T model.
                loss = self.hparams.transducer_cost(
                    logits_transducer, tokens, wav_lens, token_lens
                )
        else:
            logits_transducer, wav_lens, predicted_tokens = predictions
            # Transducer loss use logits from RNN-T model.
            loss = self.hparams.transducer_cost(
                logits_transducer, tokens, wav_lens, token_lens
            )

        if stage != sb.Stage.TRAIN:

            # Decode token terms to words
            predicted_words = self.tokenizer(
                predicted_tokens, task="decode_from_list"
            )

            # Convert indices to words
            target_words = undo_padding(tokens, token_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)


class SLU(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        if wavs.shape[1] > 68000 and stage == sb.Stage.TRAIN:
            return None
        tokens_bos, tokens_bos_lens = batch.tokens_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):

                # Apply augment
                wavs_aug = augment(wavs, wav_lens)

                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                wavs_aug_tot.append(wavs_aug)

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            wav_lens = torch.cat([wav_lens] * self.n_augment)
            tokens_bos = torch.cat([tokens_bos] * self.n_augment)

        # ASR encoder forward pass

        # # pre-trained model
        # with torch.no_grad():
        #     ASR_encoder_out = self.hparams.asr_model.encode_batch(
        #         wavs.detach(), wav_lens
        #     )

        # new model
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                print(wavs.shape, flush = True)
                feats = self.hparams.compute_features(wavs)
                feats = self.hparams.normalize(feats, wav_lens)
            del batch
            del wavs
            # print(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0), flush = True)
            ASR_encoder_out = self.hparams.enc(feats.detach())
            del feats
            # torch.cuda.synchronize()
            # gc.collect()
            #     print(ASR_encoder_out.shape)

            # SLU forward pass
            encoder_out = self.hparams.slu_enc(ASR_encoder_out)
            e_in = self.hparams.output_emb(tokens_bos)
            h, _ = self.hparams.dec_slu(e_in, encoder_out, wav_lens)

            # Output layer for seq2seq log-probabilities
            logits = self.hparams.seq_lin(h)
        else:
            with torch.no_grad():
                feats = self.hparams.compute_features(wavs)
                feats = self.hparams.normalize(feats, wav_lens)
                del batch
                ASR_encoder_out = self.hparams.enc(feats.detach())
                del feats
                # gc.collect()
                #     print(ASR_encoder_out.shape)

                # SLU forward pass
                encoder_out = self.hparams.slu_enc(ASR_encoder_out)
                e_in = self.hparams.output_emb(tokens_bos)
                h, _ = self.hparams.dec_slu(e_in, encoder_out, wav_lens)

                # Output layer for seq2seq log-probabilities
                logits = self.hparams.seq_lin(h)

        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            del ASR_encoder_out
            del encoder_out
            return p_seq, wav_lens
        else:
            p_tokens, scores = self.hparams.beam_searcher(encoder_out, wav_lens)
            del ASR_encoder_out
            del encoder_out
            del scores
            return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (NLL) given predictions and targets."""

        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.hparams, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )

        if stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos] * self.n_augment, dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens] * self.n_augment, dim=0
            )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # (No ctc loss)
        loss = loss_seq

        if (stage != sb.Stage.TRAIN) or (
            self.batch_count % show_results_every == 0
        ):
            # Decode token terms to words
            predicted_semantics = [
                tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]

            target_semantics = [wrd.split(" ") for wrd in batch.semantics]

            for i in range(len(target_semantics)):
                print(" ".join(predicted_semantics[i]).replace("|", ","))
                print(" ".join(target_semantics[i]).replace("|", ","))
                print("")

            if stage != sb.Stage.TRAIN:
                self.wer_metric.append(
                    ids, predicted_semantics, target_semantics
                )
                self.cer_metric.append(
                    ids, predicted_semantics, target_semantics
                )

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        if predictions is not None:
            loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
            loss.backward()
            if self.check_gradients(loss):
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.batch_count += 1
            return loss.detach()
        else:
            return torch.tensor(0)

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.batch_count = 0

        if stage != sb.Stage.TRAIN:

            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_train"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_valid"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides(
        "semantics", "token_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(semantics):
        yield semantics
        tokens_list = tokenizer.encode_as_ids(semantics)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data, tokenizer


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    show_results_every = 100  # plots results every N iterations

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep
    from prepare import prepare_FSC  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_FSC,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (train_set, valid_set, test_set, tokenizer,) = dataio_prepare(hparams)

    # We download and pretrain the tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Brain class initialization
    slu_brain = SLU(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    slu_brain.tokenizer = tokenizer

    # Training
    slu_brain.fit(
        slu_brain.hparams.epoch_counter,
        train_set,
        valid_set,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Test
    slu_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test.txt"
    slu_brain.evaluate(test_set, test_loader_kwargs=hparams["dataloader_opts"])
