// Run with playwright-cli run-code --filename=web/check_viewer.js against localhost:8000.
async page => {
  const assert = (ok, message) => { if (!ok) throw new Error(message); };
  const errors = [];
  const onError = error => errors.push(error.message);
  page.on('pageerror', onError);
  const originalViewport = page.viewportSize();
  const acceptNavigation = dialog => dialog.accept();
  page.on("dialog", acceptNavigation);
  const fitted = async label => {
    await page.waitForFunction(() => {
      const image = document.querySelector('#dicom-image').getBoundingClientRect();
      const stage = document.querySelector('#viewer-stage').getBoundingClientRect();
      return image.width > 0 && image.left >= stage.left + 7 && image.right <= stage.right - 7
        && image.top >= stage.top + 7 && image.bottom <= stage.bottom - 7;
    });
    assert(await page.locator('#fit-image').isDisabled(), label + ': fitted state must be explicit');
    assert(await page.locator('#fit-image').innerText() === 'Вписано', label + ': fit label');
    assert(await page.evaluate(() => {
      const i = document.querySelector('#dicom-image'), o = document.querySelector('#annotation-image');
      const a = i.getBoundingClientRect(), b = o.getBoundingClientRect();
      return Math.abs(a.width / a.height - i.naturalWidth / i.naturalHeight) < .001
        && ['x', 'y', 'width', 'height'].every(k => Math.abs(a[k] - b[k]) < .1)
        && document.documentElement.scrollWidth <= innerWidth;
    }), label + ': proportions, overlay alignment and page width');
  };
  try {
    page.setDefaultTimeout(10000);
    await page.route('**/api/health', r => r.fulfill({json:{status:'ok',busy:false}}));
    await page.goto('http://127.0.0.1:8000');
    await page.evaluate(() => {
      const shapes = [[300,317], [900,200], [120,900]];
      rows = shapes.map((_, i) => ({...Object.fromEntries(columns.map(k => [k,''])),
        path_to_study: 'проверка_длинного_имени_'.repeat(5) + i + '.dcm',
        anatomical_region: spine, quality_class:'1', quality_prob:'0.8',
        violation_type:i === 0 ? 'Неизвестное замечание' : '', processing_status:'Success'}));
      rows.push({...rows[0],path_to_study:'broken.dcm',processing_status:'Failure'});
      previews = shapes.map(([width,height]) => {
        const canvas = document.createElement('canvas'); canvas.width=width;canvas.height=height;
        const context = canvas.getContext('2d'); context.fillStyle='#142952';context.fillRect(0,0,width,height);
        context.strokeStyle='#4aa';context.lineWidth=8;context.strokeRect(4,4,width-8,height-8);
        return {image:canvas.toDataURL(),overlay:canvas.toDataURL(),width,height,legend:[]};
      });
      previews.push({message:'Повреждённый файл'}); csv=exportRows(); demo=false; selected=0;filter='all';showResults();
    });
    await page.waitForFunction(() => document.querySelector('#dicom-image').naturalWidth === 300);
    assert((await page.locator('#inspection').innerText()).includes('Неизвестное замечание'), 'Never drop an unfamiliar model finding');
    const before = await page.evaluate(() => JSON.stringify({rows,csv}));
    for (const width of [320,375,480,768,800,801,1024,1100,1101,1280,1440,1920]) {
      await page.setViewportSize({width,height:900});
      for (let index=0;index<3;index++) {
        await page.evaluate(index=>openResult(index),index);
        await page.waitForFunction(index=>document.querySelector('#dicom-image').naturalWidth === [300,900,120][index],index);
        await fitted(`${width}px image ${index}`);
      }
    }
    await page.setViewportSize({width:1440,height:900});
    await page.evaluate(()=>openResult(0));
    await page.locator('#zoom-in').click(); await page.locator('#zoom-in').click();
    const stage = await page.locator('#viewer-stage').boundingBox();
    await page.mouse.move(stage.x+stage.width/2,stage.y+stage.height/2);
    await page.mouse.down(); await page.mouse.move(stage.x+stage.width/2+60,stage.y+stage.height/2+60);await page.mouse.up();
    assert(await page.evaluate(()=>view.x!==0 || view.y!==0),'Dragging must actually pan an enlarged image');
    await page.locator('#fit-image').click(); await fitted('after pan');
    await page.locator('.viewer-settings > summary').click();
    await page.locator('#image-brightness').fill('125');await page.locator('#image-brightness').dispatchEvent('input');
    await page.locator('#zoom-in').click();await page.locator('#fit-image').click();
    assert(await page.locator('#image-brightness').inputValue()==='125','Fit must preserve brightness');
    await page.locator('#reset-image').click();
    assert(await page.locator('#image-brightness').inputValue()==='100','Reset restores brightness');
    await page.locator('.viewer-settings > summary').click();
    await page.locator('#viewer-stage').focus();await page.keyboard.press('+');await page.keyboard.press('ArrowDown');await page.keyboard.press('0');
    await fitted('keyboard reset');
    if (await page.evaluate(()=>document.fullscreenEnabled)) {
      await page.locator('#fullscreen-image').click();await page.waitForFunction(()=>!!document.fullscreenElement);
      await fitted('fullscreen');
      await page.locator('#zoom-in').click();await page.locator('#fit-image').click();await fitted('fullscreen reset');
      await page.setViewportSize({width:812,height:375});await fitted('landscape fullscreen');
      assert(await page.locator('#viewer-stage').evaluate(el=>el.getBoundingClientRect().bottom<=innerHeight),'Fullscreen image must stay on screen in landscape');
      await page.setViewportSize({width:375,height:812});await fitted('portrait fullscreen');
      for(let i=0;i<3;i++) await page.locator('#next-image').click();
      assert(await page.locator('#dicom-image').isHidden(),'Failure has no stale image');
      assert(await page.locator('#fullscreen-image').isVisible(),'Exit fullscreen remains available without a preview');
      await page.locator('#fullscreen-image').click();await page.waitForFunction(()=>!document.fullscreenElement);
    }
    await page.setViewportSize({width:375,height:812});await page.evaluate(()=>openResult(1));
    assert((await page.locator('#mobile-summary').innerText()).includes('Есть замечания модели'),'Generic positive result must not be blank');
    await page.locator('#mobile-summary .text-button').click();await page.locator('#result-row-0').click();
    assert(await page.evaluate(()=>document.activeElement.id)==='mobile-summary','Mobile list selection restores focus');
    await fitted('mobile selection');
    await page.evaluate(()=>{previews[0].overlay='';render();});
    assert(await page.locator('.annotation-toggle').isHidden(),'Do not show a dead overlay checkbox');
    await page.locator('#annotation-details > summary').click();
    assert(await page.locator('#annotation-note').isVisible(),'Unavailable overlay explanation remains reachable');
    assert(before===await page.evaluate(()=>JSON.stringify({rows,csv})),'Viewer controls must not change model output');
    assert(!errors.length,errors.join('\n'));
    return 'PASS: 36 aspect/viewport combinations, fit, pan, keyboard, brightness, overlay alignment, fullscreen failure exit, long names, generic findings, mobile focus, immutable CSV';
  } finally {
    await page.evaluate(()=>{busy=false;rows=[];});
    await page.unroute('**/api/health');
    if(await page.evaluate(()=>!!document.fullscreenElement)) await page.evaluate(()=>document.exitFullscreen());
    await page.reload();
    if(originalViewport) await page.setViewportSize(originalViewport);
    page.off("dialog", acceptNavigation);
    page.off('pageerror',onError);
  }
}
