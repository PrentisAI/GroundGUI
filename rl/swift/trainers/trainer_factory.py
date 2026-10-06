# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import inspect
from dataclasses import asdict
from typing import Dict

from swift.utils import get_logger

logger = get_logger()


class TrainerFactory:
    TRAINER_MAPPING = {
        'causal_lm': 'swift.trainers.Seq2SeqTrainer',
        'seq_cls': 'swift.trainers.Trainer',
        'embedding': 'swift.trainers.EmbeddingTrainer',
        'reranker': 'swift.trainers.RerankerTrainer',
        'generative_reranker': 'swift.trainers.RerankerTrainer',
        # rlhf
        'dpo': 'swift.rlhf_trainers.DPOTrainer',
        'orpo': 'swift.rlhf_trainers.ORPOTrainer',
        'kto': 'swift.rlhf_trainers.KTOTrainer',
        'cpo': 'swift.rlhf_trainers.CPOTrainer',
        'rm': 'swift.rlhf_trainers.RewardTrainer',
        'ppo': 'swift.rlhf_trainers.PPOTrainer',
        'grpo': 'swift.rlhf_trainers.GRPOTrainer',
        'gkd': 'swift.rlhf_trainers.GKDTrainer',
        'opsd': 'swift.rlhf_trainers.OPSDTrainer',
        'ttrt': 'swift.rlhf_trainers.TtRTTrainer',
        'opsd_grpo': 'swift.rlhf_trainers.OPSDGRPOTrainer',
    }

    TRAINING_ARGS_MAPPING = {
        'causal_lm': 'swift.trainers.Seq2SeqTrainingArguments',
        'seq_cls': 'swift.trainers.TrainingArguments',
        'embedding': 'swift.trainers.TrainingArguments',
        'reranker': 'swift.trainers.TrainingArguments',
        'generative_reranker': 'swift.trainers.TrainingArguments',
        # rlhf
        'dpo': 'swift.rlhf_trainers.DPOConfig',
        'orpo': 'swift.rlhf_trainers.ORPOConfig',
        'kto': 'swift.rlhf_trainers.KTOConfig',
        'cpo': 'swift.rlhf_trainers.CPOConfig',
        'rm': 'swift.rlhf_trainers.RewardConfig',
        'ppo': 'swift.rlhf_trainers.PPOConfig',
        'grpo': 'swift.rlhf_trainers.GRPOConfig',
        'gkd': 'swift.rlhf_trainers.GKDConfig',
        'opsd': 'swift.rlhf_trainers.OPSDConfig',
        # TtRT adds no config fields of its own (ttrt_* live on TrainArgumentsMixin),
        # so it reuses OPSDConfig rather than declaring a third empty duplicate.
        'ttrt': 'swift.rlhf_trainers.OPSDConfig',
        # OPSD+GRPO adds no config fields of its own -- nzg_* live on
        # TrainArgumentsMixin, which GRPOConfig already inherits (as it does every
        # opsd_*/ttrt_* field) -- so it reuses GRPOConfig rather than declaring a
        # duplicate. Same pattern as ttrt reusing OPSDConfig above.
        'opsd_grpo': 'swift.rlhf_trainers.GRPOConfig',
    }

    @staticmethod
    def get_cls(args, mapping: Dict[str, str]):
        if hasattr(args, 'rlhf_type'):
            train_method = args.rlhf_type
            # add
            if args.rlhf_type == 'gkd' and args.use_opsd:
                train_method = 'opsd'
            # TtRT is OPSD with a different token-weight rule. rlhf_type stays
            # 'gkd' on purpose -- that is what keeps padding_side, the beta
            # default, teacher construction, template mode, vllm_client and
            # teacher zero3 wired exactly as they are for OPSD. Checked after
            # use_opsd so ttrt wins when both flags are set.
            if args.rlhf_type == 'gkd' and getattr(args, 'use_ttrt', False):
                train_method = 'ttrt'
            # OPSD+GRPO is GRPO with a negative-zero-variance-group branch. rlhf_type
            # stays 'grpo' for the same reason ttrt keeps 'gkd': every
            # `rlhf_type == 'grpo'` check in swift/arguments/rlhf_args.py and
            # swift/pipelines/train/rlhf.py must keep firing unchanged.
            if args.rlhf_type == 'grpo' and getattr(args, 'use_opsd_grpo', False):
                train_method = 'opsd_grpo'
        else:
            train_method = args.task_type
        module_path, class_name = mapping[train_method].rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def get_trainer_cls(cls, args):
        return cls.get_cls(args, cls.TRAINER_MAPPING)

    @classmethod
    def get_training_args(cls, args):
        training_args_cls = cls.get_cls(args, cls.TRAINING_ARGS_MAPPING)
        args_dict = asdict(args)
        parameters = inspect.signature(training_args_cls).parameters

        for k in list(args_dict.keys()):
            if k not in parameters:
                args_dict.pop(k)

        args._prepare_training_args(args_dict)
        training_args = training_args_cls(**args_dict)
        return training_args
