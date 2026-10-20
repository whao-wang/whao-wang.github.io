(function () {
  const node = document.getElementById('busuanzi_page_pv');
  if (!node) return;
  const page = encodeURIComponent(location.href.split('#')[0]);
  const image = document.createElement('img');
  image.src = 'https://api.visitorbadge.io/api/visitors?path=' + page + '&label=views&countColor=%232563eb&style=flat';
  image.alt = '阅读次数';
  image.height = 20;
  image.loading = 'lazy';
  image.style.cssText = 'height:20px;vertical-align:-5px;display:inline-block';
  image.onerror = function () { node.textContent = '—'; };
  node.textContent = '';
  node.appendChild(image);
})();
