"""Aggregate runner mirroring MATLAB total_all.m.

按顺序执行各个参数扫�?脚本，便于一键复现所有图表。
"""

from __future__ import annotations

import time

from Everycom_Per_IU import main as run_everycom_per_iu
from IU_CacheSize import main as run_iu_cachesize
from IU_Computing import main as run_iu_computing
from SBS_CacheSize import main as run_sbs_cachesize
from SBS_Computing import main as run_sbs_computing
from SBS_IU_CacheSize import main as run_sbs_iu_cachesize
from SBS_IU_Computing import main as run_sbs_iu_computing
from Users_Percom import main as run_users_percom
from Zipf import main as run_zipf
from main import main as run_main


def main():
    start = time.time()
    run_everycom_per_iu()
    run_iu_cachesize()
    run_iu_computing()
    run_sbs_cachesize()
    run_zipf()
    run_main()
    run_sbs_computing()
    run_users_percom()
    run_sbs_iu_computing()
    run_sbs_iu_cachesize()
    print(f"total_all 运行完毕，总耗时 {time.time() - start:.2f} 秒")


if __name__ == "__main__":
    main()
