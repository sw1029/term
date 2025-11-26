"""
실제 실행 진입점.
왠만한 건 다 여기(main)에서 호출하되,
구체적인 실험 로직은 runner.py 내 run_experiment 가 담당한다.

huggingface 로그인이 안된 경우라면
터미널에서 huggingface cli에 토큰 등록이 필요할 수 있다.

Hydra 기반 구성:
    - config/config.yaml 를 통해 실험 설정 관리
    - main.py 는 cfg를 runner.run_experiment 에 전달하는 역할만 수행
"""

import hydra
import sys
from omegaconf import DictConfig

from runner import run_experiment


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Hydra가 넘겨주는 설정(cfg)을 받아
    run_experiment 에 위임하는 얇은 래퍼.
    """
    # 전체 실행 명령어를 문자열로 기록 (예: "python main.py ...")
    cli_command = " ".join(sys.argv)
    run_experiment(cfg, cli_command=cli_command)


if __name__ == "__main__":
    main()
