# Fate/Grand Order 技能练度截图读取工具 (草稿)

使用 Python 通过游戏截图，识别从者信息及其技能等级信息．

## 简单图片示意

可以参考 [NGA Fate/Grand Order 板块贴](https://bbs.ngacn.cc/read.php?tid=14517730&_ff=540) 的说明．

## 使用说明

### 环境

建议 Python3，Anaconda 环境．至少需要安装 numpy, matplotlib, scikit_image, jupyter 等库．

由于调试输出的图像留白很多，可以考虑安装 ImageMagick 进行命令行下的图像留白去除．

### 跑示例程序

> 执行 `servant_skill_full_process.py` 即可．

如果只是想跑我本人账号的英灵练度，可以直接在 terminal 下执行 `servant_skill_full_process.py` 的脚本．请尽量不要动目录中的其它文件．

随后会生成文件 `skill_full.csv` 与文件夹 `debug`．
* `skill_full.csv` 是你的英灵练度表格，表格中的第一列是英灵编号，可以到 `个人空间` → `卡牌图鉴` → `灵基一览` 查看．可以用 Microsoft Excel 打开．
* `debug` 文件夹下的图片直观地显示英灵编号与其对应的练度．可以查看这些图片以核实程序读取的信息是否正确．

改程序一般能在 2 分钟内跑完．

### 跑自己的英灵练度

工作会冗余一些，因为这份练度工具现在仍然不友好．

> 注意！

* 请使用 16:9 的截屏模式，并且不要让你的图片倒置或旋转储存．
* 请不要对目录 `database` 下的 `numbers`、`servant_indicator`、`servant_mask` 作改动．
* 请在国服进行操作．日服不清楚行不行，但请把窗格开到最大，因为程序定位从者与数字的依据就是靠国服的网格位置确定的．
* 请尽量用较高的清晰度截取图片，即尽量不低于 1280\*720．所有图形在程序中会统一转为 1280\*720 大小．

> 执行过程

* 首先需要截取你目前所有从者的肖像．请到 `个人空间` → `卡牌图鉴` → `灵基一览`，截取总共 14 张图片 (国服 2018-07-16 进度是 168 = 12\*14 骑从者)，截图放在 `database/my_box_20180715` 下，或者自定义文件夹，但需要在下一步对源代码进行少量修改．
* 执行 Jupyter Notebook `generate_servant_database.ipynb`．完成后，文件夹 `database/servant_database` 会储存截取下的英灵肖像．
* 随后请将包含英灵练度的截图放到你自定义的文件夹下．这种截图需要在英灵技能强化界面截取．如果你不打算更改源代码，你也可以考虑直接放在 `test_directory` 下，且这些文件的不包含后缀的文件名不能少于 2 个英文字符．
* 在 `servant_skill_full_process.py` 下的 120 -124 行作必要的改动；124 行不建议改动，因为你有必要核实这个程序是否正确运行了．也许程序还有许多需要手动调试的地方 >.< 执行上述程序．
* 如果 124 行没有改动，调试信息会储存在 `debug` 文件夹下．

## 其它文件

* `skill_data.ipynb` 叙述我如何一步一步得到最终的结果，并辅以图片说明．这个程序里面有许多参数尚未调优，但相信绝大多数做程序的思路都在里面了．
* `servant_skill_full_process.ipynb` 是一个近乎完整的程序，它读取示例文件 `database/skill/skill_7.png` 并得到最终的英灵练度信息．输出文件是 `skill_7.csv`．它是 `servant_skill_full_process.py` 的雏形，但两者可能在部分参数上不一致．
