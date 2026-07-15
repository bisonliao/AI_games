# 打字练习 Web 工具

这是一个不依赖第三方包的 Python Web 应用，用于儿童打字练习。

## 运行

```powershell
python typing_practice.py
```

默认监听 `0.0.0.0:8000`。在服务器上放行端口后，浏览器访问：

```text
http://服务器IP:8000
```

在 Windows 上也可以双击 `start_server.bat`，或在当前目录执行：

```powershell
.\start_server.bat
```

## 功能

- 首页提供初阶、中阶、高阶三个入口。
- 初阶包含 9 个指法课时，每课可反复生成随机练习。
- 中阶生成约 100 个英文单词的文章练习。
- 高阶生成约 300 个汉字的中文文章练习。
- 练习时禁止删除，正确字符显示绿色，错误字符显示红色。
- 完成后统计本次练习正确率。

## 内容库

练习内容放在 `content/` 目录：

- `content/beginner/lesson_01` 到 `lesson_09`：每课 5 个文本，每次从当前课随机抽一个。
- `content/intermediate`：20 个英文文本。
- `content/advanced`：20 个中文文本。

每次开始练习时，服务端随机读取一个文本返回给当前浏览器。练习进度只保存在浏览器端，所以多个人同时练习不会互相冲突。

如需重新生成内容库：

```powershell
python tools/generate_content_bank.py
```
