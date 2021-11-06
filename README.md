# README

## main 分支下为完整工具链

具体配置和运行方法见：

https://medium.com/@zarinlo/publish-technical-tutorials-in-google-codelab-format-b07ef76972cd

## gh-pages 分支下防止静态页面文件

即 main 分支下 /tools/site/build/ 目录中的所有文件

每次在本地测试更新内容后，对 main 分支进行更新，之后将 gh-pages 分支更新为 /tools/site/build/ 目录中的所有文件

## markdown 语法规范

参考 main 分支中的 /tools/site/codelabs/homosplitlearning.md

或 https://codelabs.solace.dev/codelabs/codelab-4-codelab/index.html?index=..%2F..index#5

注意：

*   markdown 中的图片使用相对位置，放置在 /tools/site/codelabs/assets/ 目录下
*   markdown 中应使用 **LINUX 换行符**

*   markdown 中的 List 语法经 claat export 时好像存在缩进问题