# GitHub 上传说明

仓库根目录就是本文件所在的 SIVR 文件夹。下载 SIVR_GitHub_20261026.zip 解压后，将其中 README.md、src/、configs/、scripts/、tests/、docs/、third_party/、data/aggregates/、环境文件和 .gitignore 放到 GitHub 仓库根目录。不要在仓库中再多套一层上传包目录。

推荐先建立 Private 仓库，上传本包后供老师核阅；确定作者代码许可与数据权限后再公开。第三方BraVL许可证已随目录保留；作者自己的新增代码许可证尚需决定，不能把第三方MIT声明当整个项目的授权。

先验证：python -m pip install -r requirements.txt；python scripts/reproduce_tables.py。完整科学CPU测试需安装 requirements-training.txt 中的 torch/networkx 等；执行 python -m pytest -q。

训练入口和路径参数见英文 README。这里未包含原始fMRI、图像、PCA缓存或训练检查点，也没有本轮重新跑GPU训练。当前的 environment.yml 是候选环境，requirements-cpu-tested.txt 是已验证CPU环境直接依赖，二者都不是伪造的服务器完整锁定环境。

不上传 .env、paths.local.json、runs/、results/、__pycache__/、.pytest_cache/、密码或token。ZIP已排除这些运行输出。

公开前待完成项：docs/RELEASE_CHECKLIST.md；代码来源与许可：third_party/README.md 和 docs/LICENSING.md；数据约定：docs/DATA.md；验证状态：docs/VALIDATION.md。
