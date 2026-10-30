# SSH服务器复测操作说明

本说明供手动连接SSH后使用。GitHub保存源码、协议和精选报告；数据、索引、模型、历史运行缓存不入Git。

## 1. 克隆代码

在服务器选择代码目录，然后运行：

```bash
git clone https://github.com/grapevine917/Gate.git
cd Gate
git clone --branch research/learned-gate --recurse-submodules https://github.com/grapevine917/DiskANN.git DiskANN
```

不要只克隆上游默认分支。查看`docs/deployment/PUBLICATION_VALIDATION.md`了解本次实际完成的本地验证；服务器还需自己验证，提交成功不等于服务器验证通过。

## 2. 安装依赖

建议Python 3.11或更高、C++17编译器、CMake、make、binutils（ar）、Boost program_options/container、libaio、tcmalloc、zstd、Intel MKL和Intel OpenMP。当前原生实现使用MKL头文件与ILP64接口，不应无验证直接替换成OpenBLAS。

Ubuntu的基础依赖示例（需管理员权限）：

```bash
sudo apt-get update
sudo apt-get install build-essential cmake git python3-venv libboost-program-options-dev libboost-container-dev libaio-dev libgoogle-perftools-dev libzstd-dev
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/server/requirements.txt
```

MKL/Intel OpenMP按服务器已有安装或管理员提供的方式配置。不同发行版的包名可能不同。若使用oneAPI，在安装后加载其环境；CMake也支持显式指定下述路径。无需GPU；只做冻结权重在线复测不需要安装PyTorch、scikit-learn或XGBoost训练包。

## 3. 单独迁移必要资产

在服务器为资产选择一个目录，例如`/srv/gate-assets`，保持相对于项目根的目录结构。它与代码目录、输出目录分开。例如：

```text
/srv/gate-assets/
  experiments/main_comparison/models/lr.bin
  experiments/main_comparison/data/validation.fbin
  experiments/sift1m/index_disk.index
  experiments/deep_p0/...
  experiments/sift100m_p0/...
```

完整的最小主实验输入及预期哈希见`experiments/server/assets_manifest.json`。它包括：

- 三数据集的运行时磁盘图、PQ文件，以及存在时的medoids/centroids/旋转矩阵等辅助文件。
- 每数据集lr/state/xgb/et权重与native_check/et_check核验样本。
- validation/test查询及GT；train/validation/test查询ID用于核对划分。
- 训练来源记录TRAIN_COMPLETE.json和标签校准样本。
- 每数据集一份历史L100原生返回ID，用来核验迁移后禁用Gate的搜索结果；**只使用ID，不使用历史计时**。

不需要传100M索引构建分片、原有逐查询测试缓存、日志、COMPLETE、旧虚拟环境和旧可执行文件。仅跑本清单的17维主实验，也不需要q/v分支的巨大额外向量表。不能把`ann*`全部通配复制当成最小迁移清单。

本次清单59个文件，约42.8GiB。连接信息确定后，可在本机项目根目录手动传输（替换用户名/主机，远端资产目录需已创建）：

```bash
rsync -avL --progress --files-from=experiments/server/assets_files.txt ./ 用户名@主机:/srv/gate-assets/
```

`-L`复制文件内容，避免将本地符号链接目标原样带到服务器。需要指定SSH端口时，用rsync的`-e 'ssh -p 端口'`。只传清单，不传整个230GB工作目录。

清单中大索引只有头部哈希及尺寸，不能当作完整文件校验；正式传输推荐自行生成全文件SHA256并在服务器核对。模型、查询、校准等较小文件提供全SHA256。

检查资产（仅读文件）：

```bash
.venv/bin/python experiments/server/benchmark.py inspect --asset-root /srv/gate-assets
```

该检查列出缺失文件和总尺寸；后续create还会核对发布清单的尺寸与哈希。

## 4. 创建全新运行目录

例如选择`/srv/gate-runs/server01`，必须位于代码和资产目录之外，且尚不存在：

```bash
.venv/bin/python experiments/server/benchmark.py create --asset-root /srv/gate-assets --workspace /srv/gate-runs/server01
```

它只复制必要脚本和冻结native覆盖源码，通过文件级符号链接引用资产，不复制历史实验结果。源脚本的路径适配只发生在新目录，原代码及旧结果不变。SERVER_CONTEXT.json记录源码、模型、查询和索引来源；迁移资产应保持只读。

create重复执行会拒绝覆盖；中断后的后续阶段可在同一workspace续跑。不同机器使用不同目录，不混用计时。一次只运行一套基准，避免不同工作目录并发争抢SSD；入口仅保证同workspace不会同时启动两个阶段。

## 5. 服务器重新构建与模型数值核验

```bash
.venv/bin/python experiments/server/benchmark.py build --workspace /srv/gate-runs/server01 --jobs 4
```

若CMake找不到Intel运行库，显式设置真实安装路径，例如：

```bash
.venv/bin/python experiments/server/benchmark.py build --workspace /srv/gate-runs/server01 --jobs 4 --mkl-path /opt/intel/oneapi/mkl/latest/lib/intel64 --mkl-include-path /opt/intel/oneapi/mkl/latest/include --omp-path /usr/lib/x86_64-linux-gnu
```

路径仅是示例，不保证所有服务器相同。默认在新workspace构建DiskANN静态库，再为每数据集编译其冻结搜索覆盖版本与runner，分别保留1M/100M差异。不复用本地旧二进制。`--base-build`仅供明确复用同机同源CMake构建时使用，不作为跨机器迁移方式。

每数据集自动运行LR/XGB/StateMLP/ET原生导出校验，失败会停止；结果见SERVER_BUILD.json及native_score_verification.json。校验只证明所测状态上的评分一致，还不证明启用剪枝不影响Recall。

## 6. 校准、禁用动作与固定L复测

```bash
.venv/bin/python experiments/server/benchmark.py prepare --workspace /srv/gate-runs/server01
.venv/bin/python experiments/server/benchmark.py smoke --workspace /srv/gate-runs/server01
.venv/bin/python experiments/server/benchmark.py p0a --workspace /srv/gate-runs/server01
```

prepare重新生成阈值、64配置基础网格及本机路径/索引锁；smoke检查关闭Gate/ET与原生一致、ET关闭时联合组与Gate一致、历史原生ID一致。计量保留Reader字节=Linux read_bytes=请求次数×4096及无swap检查。

### 旧版prepare重复执行时报PROTOCOL断言

首版脚本比较Python元组与JSON读回的列表，重复prepare即使参数不变也可能误报。已修复为先统一JSON表示再比较；实际配置或输入/程序哈希变化仍会拒绝，并列出差异，不覆盖旧协议。

对于已创建的旧workspace，先在Gate代码目录更新，再修补该workspace中的prepare副本：

```bash
git pull --ff-only
.venv/bin/python experiments/server/repair_prepare.py --workspace ~/gate-runs/server01
.venv/bin/python experiments/server/benchmark.py prepare --workspace ~/gate-runs/server01
```

修补器会先核验原workspace，只接受首版prepare或已修复版本，备份prepare及SERVER_CONTEXT后更新该文件的源码哈希。模型、原生程序、PROTOCOL和结果均不删除。无需create新目录、重传数据或重新build。若之后报`Existing protocol differs`，按列出的真实差异排查，不要删掉协议绕过检查。

P0-A是固定L100的5方法×15轮；Recall不一定相同。结果在workspace/experiments/p0_old_models_rerun/，不写回Git代码目录。

## 7. 扩大P0-B及严格Recall描述性补充

```bash
.venv/bin/python experiments/server/benchmark.py expanded --workspace /srv/gate-runs/server01
```

先在**本服务器**重测原64配置验证网格，再运行新增96粗搜+32验证局部细化，每方法192配置。依据验证性能选择和冻结参数后才进行正式测试；原生/启发式/ET在三个Gate实例间共享，不重复计时。

该阶段继承完整三数据集协议，随后还会运行测试质量严格匹配探测及九轮复测，可能耗时较长。不要把它当快速smoke。需要长时间运行时在tmux/screen中前台运行本命令，避免SSH断线；不要直接运行历史background.py/launch.py来猜测启动顺序。

输出分别保留validation_selected_test.csv、primary_configurations.csv、strict_results.csv及完整配置。已有查询曾被观察，严格匹配使用测试质量，仍须标描述性；更换机器不改变这一性质。

## 8. 输出与环境记录

保留SERVER_CONTEXT.json、SERVER_BUILD.json、SOURCE/SELECTION/INDEX锁、正式状态与报告，记录CPU/内存/SSD型号/文件系统/内核、编译器和库版本。建议固定CPU资源并避免其他I/O任务。默认仍为4查询线程、Beam4、10000节点缓存；改变线程/cache属于额外实验，需另建配置与报告。

主实验需要先完成上述服务器核验，本清单不宣称新服务器已编译、已运行或已复现性能。P0-C/特征分析历史代码也会发布，但它们不在此入口自动执行，不能直接拿旧脚本去消费服务器结果而不核对其独立依赖。
