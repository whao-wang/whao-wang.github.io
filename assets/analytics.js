(function () {
  const node = document.getElementById('busuanzi_page_pv');
  if (!node) return;
  const path = location.pathname.replace(/^\/+|\/+$/g, '') || 'home';
  const key = path.toLowerCase().replace(/[^a-z0-9._-]+/g, '-').slice(0, 100) || 'home';
  const endpoint = 'https://api.counterapi.dev/v1/whao-wang-blog/' + encodeURIComponent(key) + '/up';
  fetch(endpoint, { mode: 'cors', cache: 'no-store' })
    .then(function (response) {
      if (!response.ok) throw new Error('counter service returned ' + response.status);
      return response.json();
    })
    .then(function (data) {
      if (typeof data.count === 'number') node.textContent = data.count.toLocaleString('zh-CN');
      else throw new Error('invalid counter response');
    })
    .catch(function (error) {
      console.warn('阅读次数暂时不可用', error);
      node.textContent = '—';
    });
})();
