// playwright-cli -s=web-workspace run-code --filename=web/check_browser.js
async (page) => {
  const assert = (ok, message) => { if (!ok) throw new Error(message); };
  const columns = ['path_to_study', 'study_uid', 'image_uid', 'anatomical_region', 'quality_class', 'quality_prob', 'violation_type', 'processing_status', 'time_of_processing'];
  const row = (name, id, finding = false) => ({path_to_study: name, study_uid: 'study-1', image_uid: id,
    anatomical_region: 'Поясничный отдел позвоночника', quality_class: finding ? '1' : '0', quality_prob: finding ? '0.8' : '0.1',
    violation_type: finding ? 'Присутствуют посторонние предметы' : '', processing_status: 'Success', time_of_processing: '0.2'});
  const failure = {...row('broken.dcm', ''), study_uid: '', anatomical_region: '', quality_class: '', quality_prob: '', processing_status: 'Failure'};
  const pack = rows => ({rows, csv: columns.join(',') + '\n' + rows.map(r => columns.map(k => '"' + r[k].replaceAll('"', '""') + '"').join(',')).join('\n') + '\n', previews: rows.map(() => ({}))});
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
    await page.goto('http://127.0.0.1:8000');
    await upload([{name: 'scan.dcm', body: 'first'}, {name: 'scan.dcm', body: 'second'}, {name: 'broken.dcm', body: 'bad'}]);
    await page.locator('#analyze').click();
    await page.waitForFunction(() => document.querySelectorAll('.study-file').length === 3 && !document.getElementById('download').disabled);
    assert(calls.length === 3, 'Each DICOM must be sent once');
    const before = await page.evaluate(() => ({csv, rows: JSON.stringify(rows)}));
    assert(before.csv.includes('Failure'), 'Transport failure must preserve a failure row');
    await page.locator('#mark-reviewed').click();
    await page.locator('[data-filter=attention]').click();
    await page.locator('[data-filter=all]').click();
    const after = await page.evaluate(() => ({csv, rows: JSON.stringify(rows)}));
    assert(JSON.stringify(before) === JSON.stringify(after), 'Review and filters must not mutate model records or CSV');
    assert(await page.locator('.reviewed-label').count() === 1, 'Only the selected duplicate is reviewed');
    await page.evaluate(() => printReport());
    assert(await page.locator('#print-report article').count() === 3, 'Print must include every row');
    assert((await page.locator('#print-report').textContent()).includes('Версия 1.0'), 'Print must identify model version');
    await page.locator('#result-row-2').click();
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
    assert(!await page.locator('body').evaluate(el => el.classList.contains('review-mode')), 'Model page must retain its own layout');
    await page.locator('#nav-check').click();
    assert(await page.locator('.study-file').count() === 3, 'Navigation must preserve results');
    return 'PASS: multi-DICOM, partial failure, duplicate identities, immutable CSV/review, print, isolated DICOM/ZIP retry';
  } finally {
    await page.unroute('**/api/health'); await page.unroute('**/api/analyze?*');
    await page.evaluate(() => { rows = []; busy = false; });
    await page.reload();
  }
}
