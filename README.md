# Hao Wang — Personal Blog

一个无需构建工具的静态个人博客，适合直接部署到 GitHub Pages。

## 发布

1. 在 GitHub 新建公开仓库 `whao-wang.github.io`。
2. 将本目录中的全部文件推送到仓库的 `main` 分支。
3. 打开仓库 `Settings → Pages`，在 **Build and deployment** 中选择 **GitHub Actions**。
4. 等待 `Deploy static site to Pages` 工作流完成，访问 `https://whao-wang.github.io`。

## 本地预览

直接打开 `index.html`，或在本目录启动任意静态文件服务器。

## 更新文章

- 在 `posts/` 中新增 HTML 文件。
- 在 `index.html` 的 `.article-grid` 中添加文章卡片。
- 新文章必须同时显示发布日期和具体时间，统一使用北京时间，格式为 `YYYY.MM.DD HH:mm`；HTML的 `datetime` 属性使用带 `+08:00` 时区的ISO 8601格式。
- 修改 `index.html` 中的个人简介、项目和链接。
