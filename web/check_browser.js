// playwright-cli -s=web-workspace run-code --filename=web/check_browser.js
async (page) => {
  const assert = (ok, message) => { if (!ok) throw new Error(message); };
  const viewport = page.viewportSize();
  const columns = ['path_to_study', 'study_uid', 'image_uid', 'anatomical_region', 'quality_class', 'quality_prob', 'violation_type', 'processing_status', 'time_of_processing'];
  const row = (name, id, finding = false) => ({path_to_study: name, study_uid: 'study-1', image_uid: id,
    anatomical_region: ['second','two'].includes(id) ? 'Проксимальный отдел бедра' : 'Поясничный отдел позвоночника', quality_class: finding ? '1' : '0', quality_prob: finding ? '0.8' : '0.1',
    violation_type: finding ? 'Присутствуют посторонние предметы' : '', processing_status: 'Success', time_of_processing: '0.2'});
  const failure = {...row('broken.dcm', ''), study_uid: '', anatomical_region: '', quality_class: '', quality_prob: '', processing_status: 'Failure'};
  const pixel = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAFklEQVR4nGP8vyVkCwMewIRPcvgoAADUCwLLObLQTgAAAABJRU5ErkJggg==';
  const pack = rows => ({rows, csv: columns.join(',') + '\n' + rows.map(r => columns.map(k => '"' + r[k].replaceAll('"', '""') + '"').join(',')).join('\n') + '\n', previews: rows.map(r => r.processing_status === "Success" ? {image:pixel,overlay:pixel,width:8,height:8,legend:[]} : {})});
  const upload = files => page.evaluate(files => {
    const transfer = new DataTransfer();
    for (const f of files) transfer.items.add(new File([f.body], f.name, {type: f.type || 'application/dicom'}));
    const input = document.getElementById('file-input');
    input.files = transfer.files; input.dispatchEvent(new Event('change', {bubbles: true}));
  }, files);
  const calls = [];
  await page.route('**/api/health', route => route.fulfill({json: {status: 'ok', model: 'Radiance', busy: false}}));
  await page.route('**/api/analyze?*', async route => {
    const request = route.request(), url = request.url(), body = request.postData();
    calls.push({body, index: (/image_index=(\d+)/.exec(url)?.[1] || null)});
    if (body === 'null-response' || body === 'null-row') {
      await route.fulfill({body: JSON.stringify(body === 'null-response' ? null : {rows:[null],csv:'x'}), contentType:'application/json'}); return;
    }
    if (body === 'bad-preview') {
      const result = pack([row('preview.dcm', 'preview')]);
      result.previews = [{image:42,overlay:42,legend:{},annotation_notes:{}}];
      await route.fulfill({json:result}); return;
    }
    if (body === 'bad' && !await page.evaluate(() => window.testRetry)) { await route.fulfill({status: 502, json: {detail: 'Связь прервалась'}}); return; }
    if (body === 'zip') {
      const rows = url.includes('image_index=') ? [row('broken.dcm', 'recovered')]
        : [row('same.dcm', 'one', true), row('same.dcm', 'two'), failure];
      await route.fulfill({json: pack(rows)}); return;
    }
    await route.fulfill({json: pack([row(decodeURIComponent(/filename=([^&]*)/.exec(url)[1]), body, body === 'first')])});
  });
  try {
    page.setDefaultTimeout(10000);
    await page.setViewportSize({width:1440,height:1000});
    await page.goto('http://127.0.0.1:8000');
    await upload([{name:'НД_для_обучения.zip',type:'application/zip',body:'loading-state'}]);
    await page.evaluate(() => {
      window.originalXHR = window.XMLHttpRequest;
      window.XMLHttpRequest = class {
        constructor() { this.upload = {}; }
        open() {}
        setRequestHeader() {}
        send() { window.pendingUpload = this; this.upload.onload(); }
      };
    });
    await page.locator('#analyze').click();
    assert(await page.locator('.upload-actions').isHidden(), 'Inactive upload actions must not compete with processing status');
    assert((await page.locator('#drop-zone').innerText()).split('Ожидаем результат модели…').length === 2, 'Show the current stage only once');
    assert(await page.locator('#elapsed').isVisible(), 'Keep elapsed time visible while waiting');
    assert(await page.locator('#progress').evaluate(el => el === document.activeElement && !el.closest('[aria-busy="true"]')), 'Focus and announce the waiting state');
    await page.evaluate(() => {
      pendingUpload.status = 503;
      pendingUpload.responseText = JSON.stringify({detail:'Модель временно недоступна. Повторите проверку.'});
      pendingUpload.onload();
      window.XMLHttpRequest = window.originalXHR;
      delete window.originalXHR; delete window.pendingUpload;
    });
    await page.waitForFunction(() => !busy);
    assert(await page.locator('#error').isVisible() && await page.locator('#progress').isHidden(), 'A failed request must leave the waiting state');
    assert(await page.locator('#analyze').evaluate(el => el === document.activeElement && !el.disabled), 'Return focus to the retry action after failure');
    assert((await page.locator('#file-heading').innerText()) === 'НД_для_обучения.zip', 'Retain selected file after failure');
    await page.locator('#clear-file').click();
    await upload([{name: 'scan.dcm', body: 'first'}, {name: 'scan.dcm', body: 'second'}, {name: 'broken.dcm', body: 'bad'}]);
    await page.locator('#analyze').click();
    await page.waitForFunction(() => document.querySelectorAll('.study-file').length === 3 && !document.getElementById('download').disabled);
    assert(calls.length === 3, 'Each DICOM must be sent once');
    assert((await page.locator('.study-group h3').allTextContents()).join('|') === 'Позвоночник1|Бёдра1|Область не определена1', 'Group by anatomy even within one Study UID');
    const before = await page.evaluate(() => ({csv, rows: JSON.stringify(rows)}));
    await page.waitForFunction(() => document.getElementById('annotation-image').naturalWidth === 8);
    assert(!await page.locator('.criterion-help').evaluate(el => el.open), 'Guidance should start collapsed');
    assert(!await page.locator('.other-checks').evaluate(el => el.open), 'Unflagged checks should start collapsed');
    await page.locator('.criterion-help > summary').click();
    assert(await page.locator('.criterion-help p').isVisible(), 'Guidance must remain accessible');
    await page.locator('.other-checks > summary').click();
    assert(await page.locator('.other-checks .not-found').count() === 2, 'All remaining spine criteria must remain accessible');
    assert(await page.locator('#annotation-details').isVisible(), 'Overlay legend disclosure must be available');
    await page.locator('#toggle-annotations').uncheck();
    assert(await page.locator('#annotation-image').isHidden(), 'Overlay must hide independently');
    await page.locator('#toggle-annotations').check();
    await page.locator('#zoom-in').click();
    assert(await page.evaluate(() => document.getElementById('dicom-image').style.transform === document.getElementById('annotation-image').style.transform), 'Overlay and scan must share transform');
    await page.locator('#fit-image').click();
    assert(before.csv.includes('Failure'), 'Transport failure must preserve a failure row');
    await page.locator('#mark-reviewed').click();
    await page.locator('[data-filter=attention]').click();
    await page.locator('[data-filter=all]').click();
    const after = await page.evaluate(() => ({csv, rows: JSON.stringify(rows)}));
    assert(JSON.stringify(before) === JSON.stringify(after), 'Review and filters must not mutate model records or CSV');
    assert(await page.locator('.reviewed-label').count() === 1, 'Only the selected duplicate is reviewed');
    await page.evaluate(() => printReport());
    assert(await page.locator('#print-report article').count() === 3, 'Print must include every row');
    assert(await page.locator('.report-overlay').count() === 2, 'Print must retain successful-image annotations');
    assert((await page.locator('#print-report').textContent()).includes('Версия 1.0'), 'Print must identify model version');
    await page.locator('#result-row-2').click();
    assert(await page.locator('#annotation-image').isHidden(), 'Failure cannot retain another image overlay');
    assert(await page.locator('.viewer-toolbar').isHidden(), 'Image controls must hide when no preview is available');
    await page.evaluate(() => { window.testRetry = true; });
    await page.locator('#retry-file').click();
    await page.waitForFunction(() => !document.getElementById('download').disabled && !document.getElementById('retry-file'));
    const recovered = await page.evaluate(() => ({count: rows.length, ids: rows.map(r => r.image_uid), seen: reviewed.size}));
    assert(recovered.count === 3 && recovered.ids.join(',') === 'first,second,bad', 'Retry must replace only the failed row');
    assert(calls.at(-1).body === 'bad' && recovered.seen === 1, 'Retry must preserve source and previous review');
    await page.locator('#new-upload').click();
    await upload([{name: 'bundle.zip', type: 'application/zip', body: 'zip'}]);
    await page.locator('#analyze').click();
    await page.waitForFunction(() => document.querySelectorAll('.study-file').length === 3 && !document.getElementById('download').disabled);
    assert(await page.evaluate(() => csv) === pack([row('same.dcm', 'one', true), row('same.dcm', 'two'), failure]).csv, 'Original ZIP CSV must remain byte-for-byte unchanged');
    await page.locator('#result-row-2').click();
    await page.locator('#retry-file').click();
    await page.waitForFunction(() => !document.getElementById('download').disabled && !document.getElementById('retry-file'));
    assert(calls.at(-1).index === '2' && calls.at(-1).body === 'zip', 'ZIP retry must send original archive and ordinal');
    assert((await page.evaluate(() => rows.map(r => r.image_uid))).join(',') === 'one,two,recovered', 'ZIP retry must preserve duplicate identities');
    await page.locator('#nav-model').click();
    await page.locator('#model-view').waitFor({state:'visible'});
    assert(!await page.locator('body').evaluate(el => el.classList.contains('review-mode')), 'Model page must retain its own layout');
    await page.locator('.skip').focus(); await page.keyboard.press('Enter');
    assert(await page.locator('#model-view').isVisible() && await page.evaluate(()=>document.activeElement.id)==='main', 'Skip link must focus content without changing the current page');
    await page.locator('#nav-check').click();
    await page.locator('#results').waitFor({state:'visible'});
    assert(await page.locator('.study-file').count() === 3, 'Navigation must preserve results');
    await page.evaluate(pixel => {
      const base={...rows[0],quality_class:'1',violation_type:'Некорректная укладка'};
      rows=[hip,spine,hip,spine].map((anatomical_region,i)=>({...base,anatomical_region,path_to_study:`order-${i}.dcm`}));
      previews=rows.map(()=>({image:pixel,overlay:pixel,width:8,height:8,legend:[{label:'Ориентир',flagged:true}]}));
      reviewed.clear(); selected=1; filter='all'; csv=exportRows(); render();
    },pixel);
    assert((await page.locator('.study-file').evaluateAll(nodes=>nodes.map(n=>n.id))).join(',')==='result-row-1,result-row-3,result-row-0,result-row-2','Anatomical display order');
    await page.locator('#next-attention').click();
    assert(await page.evaluate(()=>selected)===3,'Next finding must follow anatomical order');
    await page.locator('#next-image').click();
    assert(await page.evaluate(()=>selected)===0,'Arrow must use the same order');
    const unchanged=await page.evaluate(()=>JSON.stringify({rows,csv}));
    await page.evaluate(()=>{previews[0].overlay='data:image/png;base64,bm90LXBuZw==';render();});
    await page.waitForFunction(()=>!previews[0].overlay);
    await page.locator('#mark-reviewed').click();
    assert(await page.locator('#annotation-image').isHidden(),'Broken overlay must stay hidden after render');
    assert(await page.locator('#annotation-legend').textContent()==='','No stale overlay legend');
    assert(!(await page.locator('#inspection').textContent()).includes('Оранжевая разметка'),'No stale color explanation');
    await page.evaluate(()=>{previews[0].image='data:image/png;base64,bm90LXBuZw==';render();});
    await page.waitForFunction(()=>!previews[0].image);
    await page.locator('#mark-reviewed').click();
    assert(await page.locator('#dicom-image').isHidden(),'Broken source must stay hidden after render');
    assert((await page.locator('#viewer-message').textContent()).includes('Не удалось'),'Image failure message must persist');
    assert(unchanged===await page.evaluate(()=>JSON.stringify({rows,csv})),'Decode failures must preserve model results');
    await page.setViewportSize({width:375,height:812});
    await page.locator('#mobile-summary .text-button').click();
    await page.locator('[data-filter=failure]').click();
    await page.locator('#study-list-summary').click();
    assert(await page.locator('#mobile-summary .text-button').isVisible(),'Empty mobile filter must retain a way back to the list');
    await page.locator('#mobile-summary .text-button').click();
    await page.locator('[data-filter=all]').click();
    await page.locator('#result-row-0').click();
    assert(await page.evaluate(()=>document.activeElement.id)==='mobile-summary','Mobile selection must focus its summary');
    for (const body of ['null-response','null-row','bad-preview']) {
      await page.locator('#new-upload').click();
      await upload([{name:'response.dcm',body}]); await page.locator('#analyze').click();
      await page.waitForFunction(()=>rows.length===1 && !busy);
      assert(await page.evaluate(()=>rows[0].processing_status)===(body==='bad-preview'?'Success':'Failure'),'Malformed replies must settle the request without losing valid model rows');
      assert(await page.locator('#dicom-image').isHidden(),'Invalid preview data must never reach the image viewer');
      assert(!await page.locator('#download').isDisabled(),'Recovered response must remain downloadable');
    }
    return 'PASS: upload, retries, anatomy/navigation, overlay/zoom/print, decode failures, immutable CSV, mobile empty filter, skip link, malformed responses';
  } finally {
    await page.unroute('**/api/health'); await page.unroute('**/api/analyze?*');
    await page.evaluate(() => { rows = []; busy = false; });
    await page.reload();
    if (viewport) await page.setViewportSize(viewport);
  }
}
