"""Evaluate one policy with the same real LIBERO evaluator; never train it."""

import os


def evaluate_once(self):
    try:
        if self.cfg.runner.get('only_eval', False):
            raise RuntimeError('Portable eval requires runner.only_eval=false for symmetric weight-sync initialization')
        self.global_step = int(os.environ['SPATIAL_REPRO_EVAL_STEP'])
        self.update_rollout_weights()
        result = self.evaluate()
        self.metric_logger.log(data={f'eval/{key}': value for key, value in result.items()}, step=self.global_step)
    finally:
        self._finish_run()


def main():
    from spatial_wovr_eval import SpatialEvaluationMixin
    SpatialEvaluationMixin.run = evaluate_once
    from train_spatial_wovr import main as train_driver
    return train_driver()


if __name__ == '__main__':
    main()
