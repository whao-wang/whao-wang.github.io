document.addEventListener('DOMContentLoaded', () => {
  const article = document.querySelector('.post-content');
  if (!article) return;

  const section = document.createElement('section');
  section.className = 'comments';
  section.setAttribute('aria-labelledby', 'comments-title');
  section.innerHTML = `
    <p class="eyebrow">COMMENTS</p>
    <h2 id="comments-title">一起聊聊</h2>
    <p class="comments-intro">欢迎留下你的想法。评论使用 GitHub 登录，并保存在本站的公开 Issues 中。</p>
  `;

  const widget = document.createElement('script');
  widget.src = 'https://utteranc.es/client.js';
  widget.setAttribute('repo', 'whao-wang/whao-wang.github.io');
  widget.setAttribute('issue-term', 'pathname');
  widget.setAttribute('theme', 'preferred-color-scheme');
  widget.setAttribute('crossorigin', 'anonymous');
  widget.async = true;

  section.appendChild(widget);
  article.appendChild(section);
});
