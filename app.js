(() => {
  'use strict';
  const svg = document.getElementById('polar-map');
  const NS = 'http://www.w3.org/2000/svg';
  const state = { day: 3, vesselProgress: 3, horizon: 3, route: 'balanced', layers: { ice: true, routes: true, bergs: true }, compare: true, zoom: 1, playing: false };
  let currentScenarioPaths=null;
  let replayFrame=null;
  const grid = { x0: 186, y0: 65, dx: 18, dy: 18, cols: 44, rows: 34 };
  const defaultOrigin = { x: 270, y: 496 }, defaultDestination = { x: 706, y: 281 };
  const origin = { ...defaultOrigin }, destination = { ...defaultDestination };
  let originLabel='Rothera', destinationLabel='Halley VI', endpointPick=null;
  const environmentDay=()=>state.day+state.horizon;
  // Approximate coastal outline used by the drawing and by route cell rejection.
  const landShape = [[285,94],[316,79],[346,92],[369,83],[395,71],[412,98],[437,87],[458,78],[480,99],[502,86],[529,70],[552,93],[574,82],[602,67],[630,93],[651,101],[680,111],[724,131],[753,160],[771,181],[769,213],[752,234],[730,251],[710,249],[693,267],[677,287],[658,298],[640,300],[616,301],[607,284],[590,290],[568,298],[551,314],[534,308],[514,301],[506,280],[486,283],[466,286],[453,305],[436,297],[418,288],[409,267],[390,270],[370,273],[364,288],[347,277],[327,264],[321,249],[303,243],[282,235],[269,218],[273,197],[277,175],[260,152],[267,131],[271,116],[276,103]];
  const baseBergs = [
    { id:'A23a', x:534, y:356, driftX:5.1, driftY:-2.7, radius:25 },
    { id:'B09B', x:619, y:295, driftX:3.0, driftY:-1.3, radius:18 },
    { id:'D15A', x:405, y:251, driftX:2.2, driftY:1.5, radius:15 },
    { id:'C19A', x:604, y:466, driftX:-2.2, driftY:-.4, radius:13 }
  ];
  const routeStyles = {
    balanced: { label:'Balanced', color:'#8cded1', ice:3.8, berg:8.5 },
    fast: { label:'Shortest', color:'#e2bd8c', ice:.8, berg:3.2 },
    safer: { label:'Lower ice', color:'#9bbfe0', ice:8.3, berg:14 }
  };
  const el = (tag, attrs={}, parent=svg, text='') => { const n=document.createElementNS(NS,tag); for(const [k,v] of Object.entries(attrs)) n.setAttribute(k,v); if(text) n.textContent=text; parent.appendChild(n); return n; };
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
  function isLand(p){let inside=false;for(let i=0,j=landShape.length-1;i<landShape.length;j=i++){const [xi,yi]=landShape[i],[xj,yj]=landShape[j];const crosses=(yi>p.y)!==(yj>p.y)&&p.x<(xj-xi)*(p.y-yi)/(yj-yi)+xi;if(crosses)inside=!inside;}return inside;}
  const smooth=(a,b,x)=>{const t=clamp((x-a)/(b-a),0,1);return t*t*(3-2*t)};
  const cellPoint=(c,r)=>({x:grid.x0+c*grid.dx+grid.dx/2,y:grid.y0+r*grid.dy+grid.dy/2});
  const toCell=p=>({c:clamp(Math.round((p.x-grid.x0-grid.dx/2)/grid.dx),0,grid.cols-1),r:clamp(Math.round((p.y-grid.y0-grid.dy/2)/grid.dy),0,grid.rows-1)});
  function bergs(){const day=environmentDay();return baseBergs.map(b=>({...b,x:b.x+b.driftX*day*.72,y:b.y+b.driftY*day*.72}));}
  function iceAt(x,y,day=environmentDay(),observed=false){
    const shift=observed?-8:day*1.15;
    const patch1=Math.exp(-(((x-(478+shift))/153)**2+((y-309)/125)**2)*1.35);
    const patch2=.68*Math.exp(-(((x-(626+shift*.55))/110)**2+((y-406)/156)**2)*1.22);
    const patch3=.43*Math.exp(-(((x-375)/100)**2+((y-211+day*1.4)/84)**2)*1.55);
    const wave=.08*Math.sin((x+y*.58+day*13)/41)+.06*Math.cos((x*.7-y+day*9)/37);
    const opening=.24*Math.exp(-(((y-(434-(x-280)*.48))/31)**2));
    const v=4+88*(.56*patch1+.52*patch2+.32*patch3)+wave*100-opening*62+(observed?7*Math.sin((x-y)/49):0);
    return clamp(v,0,100);
  }
  function fieldColor(v){
    if(v<10)return '#0b2b3b'; if(v<25)return '#15506b'; if(v<40)return '#377389'; if(v<55)return '#5e9bac'; if(v<70)return '#93bec6'; if(v<84)return '#c6dfe0'; return '#e2eef0';
  }
  function addDefs(){
    const defs=el('defs');
    let clip=el('clipPath',{id:'polar-clip'},defs);el('circle',{cx:480,cy:356,r:300},clip);
    const vesselClip=el('clipPath',{id:'vessel-cutout',clipPathUnits:'userSpaceOnUse'},defs);
    el('path',{d:'M0 -47 C9 -39 12 -27 14 -16 L17 13 Q18 31 13 42 Q7 47 0 47 Q-7 47 -13 42 Q-18 31 -17 13 L-14 -16 C-12 -27 -9 -39 0 -47Z'},vesselClip);
    const g=el('linearGradient',{id:'uncertainty-gradient',x1:'0',x2:'0',y1:'0',y2:'1'},defs);el('stop',{offset:'0%','stop-color':'#d0e8ed','stop-opacity':'.17'},g);el('stop',{offset:'100%','stop-color':'#d0e8ed','stop-opacity':'.02'},g);
  }
  function drawGrid(){
    const g=el('g',{id:'ice-layer','clip-path':'url(#polar-clip)',opacity:state.layers.ice?'.82':'.0'});
    for(let r=0;r<grid.rows;r++)for(let c=0;c<grid.cols;c++){
      const p=cellPoint(c,r);if(dist(p,{x:480,y:356})>299)continue;
      const v=iceAt(p.x,p.y);el('rect',{x:p.x-grid.dx/2,y:p.y-grid.dy/2,width:grid.dx+1,height:grid.dy+1,fill:fieldColor(v),opacity:state.compare?.78:.8},g);
    }
    // forecast uncertainty band at the moving ice edge
    if(state.layers.ice){const day=environmentDay();const u=el('path',{d:`M320 ${445+day} C390 ${395+day*2}, 454 ${420-day}, 525 ${370-day*2} S646 ${338-day}, 686 ${275-day}`,fill:'none',stroke:'#d3e9ed','stroke-width':state.horizon*3.5+7,'stroke-opacity':'.10','stroke-dasharray':'2 8'},g);u.setAttribute('stroke-linecap','round');}
    if(state.compare){
      const obs=el('g',{opacity:'.48','clip-path':'url(#polar-clip)'});
      for(let r=0;r<grid.rows;r+=2)for(let c=0;c<grid.cols;c+=2){const p=cellPoint(c,r);if(dist(p,{x:480,y:356})>299)continue;let v=iceAt(p.x,p.y,environmentDay(),true);if(Math.abs(v-iceAt(p.x,p.y))>12)el('circle',{cx:p.x,cy:p.y,r:2,fill:'#d8eaf0'},obs);}
    }
    return g;
  }
  function drawGeography(){
    const g=el('g',{fill:'none','stroke-linecap':'round'});
    el('circle',{cx:480,cy:356,r:300,stroke:'#8fb3bd','stroke-opacity':'.42','stroke-width':'1'},g);
    [78,150,223].forEach(r=>el('circle',{cx:480,cy:356,r,stroke:'#9cb9c0','stroke-opacity':'.16','stroke-width':'.7','stroke-dasharray':'3 7'},g));
    for(let a=0;a<360;a+=30){let rad=a*Math.PI/180;el('path',{d:`M480 356 L${480+300*Math.cos(rad)} ${356+300*Math.sin(rad)}`,stroke:'#9cb9c0','stroke-opacity':'.14','stroke-width':'.7'},g);}
    // Antarctic land mass and shelf edge
    const land=el('path',{d:'M285 94 C316 79 346 92 369 83 C395 71 412 98 437 87 C458 78 480 99 502 86 C529 70 552 93 574 82 C602 67 630 93 651 101 C680 111 724 131 753 160 C771 181 769 213 752 234 C730 251 710 249 693 267 C677 287 658 298 640 300 C616 301 607 284 590 290 C568 298 551 314 534 308 C514 301 506 280 486 283 C466 286 453 305 436 297 C418 288 409 267 390 270 C370 273 364 288 347 277 C327 264 321 249 303 243 C282 235 269 218 273 197 C277 175 260 152 267 131 C271 116 276 103 285 94Z',fill:'#bed7df',stroke:'#ecf5f4','stroke-opacity':'.72','stroke-width':'1.2'});
    land.setAttribute('filter','');
    // shelf fractures
    const cracks=el('g',{fill:'none',stroke:'#628b9a','stroke-opacity':'.38','stroke-width':'1'});
    ['M340 134 Q377 146 408 125','M430 120 Q470 140 493 112','M534 115 Q566 141 607 120','M315 187 Q359 177 381 206','M579 179 Q624 162 663 189','M444 225 Q483 210 515 236'].forEach(d=>el('path',{d},cracks));
    // geographic labels
    const labs=el('g',{fill:'#a9c4cc','font-family':'DM Mono, monospace','font-size':'9','letter-spacing':'1'});
    el('text',{x:485,y:122,'text-anchor':'middle',fill:'#3d6877','font-size':'8'},labs,'ANTARCTIC PENINSULA');
    el('text',{x:515,y:545,'text-anchor':'middle',fill:'#9ebdc7','font-size':'8'},labs,'BELLINGSHAUSEN SEA');
    el('text',{x:727,y:399,'text-anchor':'middle',fill:'#9ebdc7','font-size':'8',transform:'rotate(75 727 399)'},labs,'WEDDELL SEA');
    el('text',{x:306,y:379,'text-anchor':'middle',fill:'#9ebdc7','font-size':'8',transform:'rotate(-72 306 379)'},labs,'AMUNDSEN SEA');
    const graticule=el('g',{fill:'#aac7ce','font-family':'DM Mono, monospace','font-size':'7','opacity':'.65'});
    [['70°S',555,361],['72°S',607,361],['74°S',656,361]].forEach(a=>el('text',{x:a[1],y:a[2]},graticule,a[0]));
  }
  function astar(style,bergList){
    const start=toCell(origin),goal=toCell(destination),key=(c,r)=>`${c},${r}`;
    let open=[{c:start.c,r:start.r,g:0,f:0}],came=new Map(),best=new Map([[key(start.c,start.r),0]]),closed=new Set();
    const dirs=[[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]];
    while(open.length){open.sort((a,b)=>a.f-b.f);let cur=open.shift(),ck=key(cur.c,cur.r);if(closed.has(ck))continue;if(cur.c===goal.c&&cur.r===goal.r){let path=[cur];while(came.has(key(path[0].c,path[0].r)))path.unshift(came.get(key(path[0].c,path[0].r)));return path.map((p,i,a)=>i===0?{...origin}:i===a.length-1?{...destination}:cellPoint(p.c,p.r));}closed.add(ck);
      for(const [dc,dr] of dirs){const c=cur.c+dc,r=cur.r+dr;if(c<0||r<0||c>=grid.cols||r>=grid.rows)continue;const p=cellPoint(c,r);if(dist(p,{x:480,y:356})>294||isLand(p))continue;const near=Math.min(...bergList.map(b=>dist(p,b)-b.radius));if(near<6)continue;
        const concentration=iceAt(p.x,p.y);const bergRisk=bergList.reduce((s,b)=>s+Math.exp(-Math.max(0,dist(p,b)-b.radius)/40),0);const iceRisk=Math.pow(concentration/100,1.65);const step=Math.hypot(dc,dr)*(1+style.ice*iceRisk+style.berg*bergRisk*.35);
        const nk=key(c,r),ng=cur.g+step;if(ng<(best.get(nk)??Infinity)){best.set(nk,ng);came.set(nk,cur);const heuristic=Math.hypot(goal.c-c,goal.r-r);open.push({c,r,g:ng,f:ng+heuristic});}
      }
    }
    return [origin,destination];
  }
  function smoothPath(points){if(points.length<3)return points.map(p=>`${p.x},${p.y}`).join(' ');let d=`M${points[0].x} ${points[0].y}`;for(let i=1;i<points.length-1;i++){let mid={x:(points[i].x+points[i+1].x)/2,y:(points[i].y+points[i+1].y)/2};d+=` Q${points[i].x} ${points[i].y} ${mid.x} ${mid.y}`;}let z=points.at(-1);return d+` T${z.x} ${z.y}`;}
  function routeStats(points,bergList){let n=0,ice=0,min=Infinity;for(let i=0;i<points.length;i++){let p=points[i];if(i%3===0){ice+=iceAt(p.x,p.y);n++;}for(const b of bergList)min=Math.min(min,dist(p,b)-b.radius);}const km=points.reduce((s,p,i)=>i?s+dist(p,points[i-1])*1.42:s,0);return{nm:Math.round(km*.54),mean:n?ice/n:0,min:Math.max(0,min*1.42*.54),hours:Math.round(km*.54/11.5)};}
  function pointAlongPath(points,fraction){
    if(!points?.length)return{x:origin.x,y:origin.y,angle:0};
    if(points.length===1)return{...points[0],angle:0};
    const segments=points.slice(1).map((point,index)=>dist(point,points[index]));
    const total=segments.reduce((sum,length)=>sum+length,0);
    let remaining=clamp(fraction,0,1)*total;
    for(let index=0;index<segments.length;index++){
      const length=segments[index];
      if(remaining<=length||index===segments.length-1){const a=points[index],b=points[index+1],t=length?remaining/length:0;return{x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t,angle:Math.atan2(b.y-a.y,b.x-a.x)*180/Math.PI+90};}
      remaining-=length;
    }
    return{...points.at(-1),angle:0};
  }
  function updateVesselPosition(){
    const point=pointAlongPath(currentScenarioPaths?.[state.route]?.pts,state.vesselProgress/7);
    const marker=document.getElementById('vessel-marker');
    if(marker)marker.setAttribute('transform',`translate(${point.x.toFixed(2)} ${point.y.toFixed(2)}) rotate(${point.angle.toFixed(2)})`);
    const day=Math.floor(state.vesselProgress),percent=Math.round(state.vesselProgress/7*100);
    document.getElementById('vessel-map-status').textContent=`Day ${day} / 7 · ${percent}% along path`;
    document.getElementById('timeline-vessel-status').textContent=`Day ${day} / 7 · ${percent}% complete`;
    const timeline=document.getElementById('timeline');timeline.style.setProperty('--timeline-progress',`${clamp(state.vesselProgress/7*100,0,100)}%`);
  }
  function drawVessel(){
    const marker=el('g',{id:'vessel-marker',role:'img','aria-label':'Illustrative research vessel in a simulated replay; no live tracking','pointer-events':'none'});
    el('title',{},marker,'Illustrative research vessel — simulated route replay, not a live position');
    el('circle',{r:24,fill:'#082637','fill-opacity':'.83',stroke:'#b8e5e7','stroke-opacity':'.68','stroke-width':1},marker);
    el('image',{href:'indian-research-vessel.png',x:-31,y:-47,width:62,height:94,'preserveAspectRatio':'xMidYMid meet','clip-path':'url(#vessel-cutout)','aria-hidden':'true'},marker);
    updateVesselPosition();
  }
  function drawRoutes(bergList){
    const g=el('g',{id:'route-layer',opacity:state.layers.routes?1:0,'clip-path':'url(#polar-clip)'});const paths={};
    // Calculate and draw less prominent alternatives first.
    for(const name of ['fast','safer','balanced']){const pts=astar(routeStyles[name],bergList);const st=routeStats(pts,bergList);paths[name]={pts,stats:st};const selected=name===state.route;const d=smoothPath(pts);if(selected)el('path',{d,fill:'none',stroke:'#082637','stroke-width':5.3,'stroke-opacity':'.72','stroke-linecap':'round','stroke-linejoin':'round'},g);el('path',{d,fill:'none',stroke:routeStyles[name].color,'stroke-width':selected?3.3:1.8,'stroke-opacity':selected?'.97':'.7','stroke-dasharray':selected?'':'5 5','stroke-linecap':'round','stroke-linejoin':'round'},g);}
    // Endpoints and selected line markers
    el('circle',{cx:origin.x,cy:origin.y,r:7,fill:'#0b262d',stroke:'#83ddc0','stroke-width':2},g);el('circle',{cx:origin.x,cy:origin.y,r:2.5,fill:'#9be4c8'},g);
    el('circle',{cx:destination.x,cy:destination.y,r:7,fill:'#0b262d',stroke:'#f0c17f','stroke-width':2},g);el('path',{d:`M${destination.x-3} ${destination.y} l3 -4 l3 4 l-3 4 Z`,fill:'#f0c17f'},g);
    const text=el('g',{fill:'#d1dfd8','font-family':'DM Mono, monospace','font-size':'8','letter-spacing':'.5'});el('text',{x:origin.x-8,y:origin.y+22,'text-anchor':'middle'},text,originLabel.toUpperCase());el('text',{x:destination.x+5,y:destination.y-12,'text-anchor':'middle'},text,destinationLabel.toUpperCase());
    return paths;
  }
  function drawBergs(list){
    const g=el('g',{id:'berg-layer',opacity:state.layers.bergs?1:0});
    for(const b of list){const spread=11+state.horizon*2;el('ellipse',{cx:b.x,cy:b.y,rx:b.radius+spread,ry:b.radius*.55+spread*.65,transform:`rotate(-24 ${b.x} ${b.y})`,fill:'url(#uncertainty-gradient)',stroke:'#85c8c5','stroke-width':'1','stroke-opacity':'.45','stroke-dasharray':'3 4'},g);el('circle',{cx:b.x,cy:b.y,r:Math.max(3,b.radius*.25),fill:'#b6e5dc',stroke:'#e4f3e9','stroke-width':'1'},g);el('path',{d:`M${b.x-18} ${b.y+18} L${b.x-6} ${b.y+6}`,stroke:'#9dccc5','stroke-width':'1','stroke-dasharray':'2 2'},g);el('text',{x:b.x+8,y:b.y-8,fill:'#aecdc8','font-family':'DM Mono, monospace','font-size':'7'},g,b.id);}
  }
  function updateStats(paths){
    for(const name of Object.keys(routeStyles)){const n=document.getElementById(`${name}-distance`);if(n)n.textContent=paths[name].stats.nm.toLocaleString();}
    const s=paths[state.route].stats;document.getElementById('transit-value').textContent=`~${s.hours} h`;document.getElementById('exposure-value').textContent=`${Math.round(s.mean)}% avg`;document.getElementById('buffer-value').textContent=`${Math.round(s.min)} NM`;
    const avg=s.mean;document.getElementById('ice-value').textContent=`${Math.round(avg)}%`;document.getElementById('ice-meter').style.width=`${clamp(avg,3,100)}%`;document.getElementById('ice-label').textContent=avg<28?'Mostly open water':avg<55?'Variable ice conditions':'Dense ice in corridor';
    document.getElementById('edge-note').textContent=`Illustrative edge shifts with the +${state.horizon} day horizon`;
    const d=new Date(Date.UTC(2025,5,15+state.day+state.horizon));document.getElementById('forecast-date').textContent=d.toLocaleDateString('en-GB',{day:'2-digit',month:'short',timeZone:'UTC'}).toUpperCase();
    document.getElementById('timeline-date').textContent=new Date(Date.UTC(2025,5,15+state.day)).toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric',timeZone:'UTC'}).toUpperCase();document.getElementById('timeline-window').textContent=`+${state.horizon} days`;
    const ex=document.getElementById('route-explanation');const messages={balanced:`Trades ${s.nm.toLocaleString()} NM of transit against ${Math.round(s.mean)}% average corridor ice in this scenario.`,fast:`Favors distance and time. Ice exposure is higher along parts of this illustrative passage.`,safer:`Weights dense ice and iceberg proximity more heavily, adding distance to reduce modeled exposure.`};ex.querySelector('span:last-child').textContent=messages[state.route];
    document.querySelectorAll('.route-option').forEach(b=>b.classList.toggle('selected',b.dataset.route===state.route));
    document.getElementById('route-endpoints-label').textContent=`${originLabel} → ${destinationLabel}`;
    document.getElementById('endpoint-status').textContent=`${originLabel} to ${destinationLabel}`;
  }
  function render(){svg.innerHTML='';addDefs();drawGrid();drawGeography();const bergList=bergs();const paths=drawRoutes(bergList);currentScenarioPaths=paths;drawBergs(bergList);drawVessel();updateStats(paths);svg.classList.remove('map-enter');void svg.getBoundingClientRect();svg.classList.add('map-enter');}
  const apiRoot=()=>{
    const configured=window.SOUTHERN_PASSAGE_API_URL;
    const localHost=['localhost','127.0.0.1','::1'].includes(window.location.hostname);
    const developmentOverride=localHost?new URLSearchParams(window.location.search).get('api'):null;
    const defaultRoot=localHost&&!window.location.pathname.startsWith('/portal')?'http://127.0.0.1:8000':window.location.origin;
    return String(configured||developmentOverride||defaultRoot).replace(/\/$/,'');
  };
  let observationCatalog=null;
  let observedView=false;
  let observationRequestId=0;
  let currentObservedDate=null;
  let currentObservationData=null;
  let researchForecastRequestId=0;
  let researchForecastController=null;
  let forecastView=false;
  let routeExposureRequestId=0;
  let routeExposureController=null;
  let routeReviewInput=null;
  let routeReviewSourceHash=null;
  let routeReviewBundle=null;
  let routeReviewBundlePromise=null;
  let routeExportController=null;
  let routeCandidateRequestId=0;
  let routeCandidateController=null;
  function setSvgHidden(element,hidden){if(hidden)element.setAttribute('hidden','');else element.removeAttribute('hidden');}
  async function getJson(url,options={}){
    const controller=new AbortController();
    const externalSignal=options.signal;
    const abort=()=>controller.abort();
    if(externalSignal?.aborted)controller.abort();
    else externalSignal?.addEventListener('abort',abort,{once:true});
    const timeout=setTimeout(()=>controller.abort(),options.timeoutMs||8000);
    try{const {timeoutMs,signal,...request}=options;const response=await fetch(url,{...request,credentials:'include',signal:controller.signal,headers:{Accept:'application/json',...(request.headers||{})}});if(!response.ok){const body=await response.json().catch(()=>({}));const error=new Error(body.detail||`API returned ${response.status}`);error.status=response.status;throw error;}return await response.json();}
    finally{clearTimeout(timeout);externalSignal?.removeEventListener('abort',abort);}
  }
  function formatTrackCoordinate(value,positive,negative){
    if(!Number.isFinite(value))return 'Position unavailable';
    return `${Math.abs(value).toFixed(1)}°${value<0?negative:positive}`;
  }
  function formatTrackDate(value){
    if(!/^\d{4}-\d{2}-\d{2}$/.test(String(value||'')))return 'date unavailable';
    return new Date(`${value}T00:00:00Z`).toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric',timeZone:'UTC'});
  }
  function renderObservedIcebergTracks(report){
    const list=document.getElementById('iceberg-track-list');
    const count=document.getElementById('iceberg-track-count');
    const status=document.getElementById('iceberg-track-status');
    const features=Array.isArray(report.features)?report.features:[];
    const asOf=report.query?.as_of_date;
    list.replaceChildren();
    count.textContent=`${features.length} TRACKS`;
    if(!features.length){status.textContent=`No source-measured iceberg positions were reported for the ${report.query?.window_days||90}-day archive window ending ${asOf||'the selected date'}.`;return;}
    const ordered=[...features].sort((a,b)=>Number(a.properties?.age_days)-Number(b.properties?.age_days)||String(a.id).localeCompare(String(b.id)));
    status.textContent=`${formatTrackDate(report.query?.window_start)} to ${formatTrackDate(asOf)} · 90-day archive window`;
    for(const feature of ordered.slice(0,3)){
      const coords=feature.geometry?.coordinates;
      const point=feature.geometry?.type==='Point'?coords:Array.isArray(coords)?coords.at(-1):null;
      const lon=Array.isArray(point)?Number(point[0]):NaN;
      const lat=Array.isArray(point)?Number(point[1]):NaN;
      const row=document.createElement('div');row.className='berg-row';row.setAttribute('role','listitem');
      const marker=document.createElement('span');marker.className='berg-marker';marker.setAttribute('aria-hidden','true');
      const name=document.createElement('div');name.className='berg-name';
      const id=document.createElement('b');id.textContent=String(feature.properties?.iceberg_id||feature.id||'Unknown iceberg');
      const date=document.createElement('small');date.textContent=`Last measured · ${formatTrackDate(feature.properties?.last_observed_date)}`;
      name.append(id,date);
      const location=document.createElement('span');location.className='berg-distance';
      const coordinates=document.createElement('b');coordinates.textContent=`${formatTrackCoordinate(lat,'N','S')} · ${formatTrackCoordinate(lon,'E','W')}`;
      const age=document.createElement('small');const ageDays=Number(feature.properties?.age_days);age.textContent=Number.isFinite(ageDays)?(ageDays===0?'At archive cutoff':`${ageDays} d before cutoff`):'Age unavailable';
      location.append(coordinates,age);row.append(marker,name,location);list.append(row);
    }
  }
  async function loadObservedIcebergTracks(){
    const count=document.getElementById('iceberg-track-count');
    const status=document.getElementById('iceberg-track-status');
    try{
      const query=new URLSearchParams({as_of_date:'2025-04-22',window_days:'90'});
      const report=await getJson(`${apiRoot()}/api/v1/icebergs/observed-tracks?${query}`,{timeoutMs:12000});
      if(report.status!=='historical_observations_only'||report.dataset?.published_coverage_end!=='2025-04-22')throw new Error('Unexpected archive provenance.');
      renderObservedIcebergTracks(report);
    }catch(error){
      count.textContent='UNAVAILABLE';
      status.textContent=error.status===401||error.status===403?'Sign in through the approved ministry gateway to view archive observations.':error.status===503?'The archive could not be verified. Check its read-only data mount and checksum manifest.':'Historical tracks are unavailable. Start the API with the verified BYU/NIC archive mounted, then retry.';
    }
  }
  function clearRouteExposure(keepReviewOpen=false){
    routeExposureRequestId++;
    routeExposureController?.abort();routeExposureController=null;
    routeExportController?.abort();routeExportController=null;routeReviewInput=null;routeReviewSourceHash=null;routeReviewBundle=null;routeReviewBundlePromise=null;
    window.dispatchEvent(new Event('southernpassage:route-cleared'));
    routeCandidateRequestId++;
    routeCandidateController?.abort();routeCandidateController=null;
    document.getElementById('route-screen-result').hidden=true;
    const exportButton=document.getElementById('route-export-button');exportButton.disabled=false;exportButton.textContent='Download review packet';
    const geoButton=document.getElementById('route-geojson-button');geoButton.disabled=false;geoButton.textContent='Download map data';
    document.getElementById('route-export-status').textContent='Includes source hashes and a human-review status.';
    document.getElementById('route-screen-error').hidden=true;
    document.getElementById('route-candidate-results').hidden=true;
    document.getElementById('route-candidate-error').hidden=true;
    setSvgHidden(document.getElementById('observation-route-overlay'),true);
    setSvgHidden(document.getElementById('route-candidate-overlay'),true);
    document.getElementById('map-canvas').removeAttribute('data-route-visible');
    document.getElementById('map-canvas').removeAttribute('data-context-band-visible');
    document.getElementById('map-canvas').removeAttribute('data-candidates-visible');
    document.getElementById('observation-route-path').setAttribute('d','');
    document.getElementById('observation-context-band').setAttribute('d','');
    ['candidate-path-shortest','candidate-path-tradeoff','candidate-path-low-ice'].forEach(id=>document.getElementById(id).setAttribute('d',''));
    const button=document.getElementById('route-screen-submit');button.disabled=false;button.textContent='Review observed exposure';button.removeAttribute('aria-busy');
    const candidateButton=document.getElementById('route-candidate-submit');candidateButton.disabled=false;candidateButton.textContent='Compare observed paths';candidateButton.removeAttribute('aria-busy');
    document.querySelector('.map-legend .legend-item:nth-of-type(2) span').textContent='Reviewed centerline';
    if(!keepReviewOpen)document.querySelector('.route-secondary-review').open=false;
  }
  function previousObservationDate(day){
    const previous=new Date(`${day}T00:00:00Z`);
    if(!Number.isFinite(previous.valueOf()))return null;
    previous.setUTCDate(previous.getUTCDate()-1);
    return previous.toISOString().slice(0,10);
  }
  function updateForecastToggle(){
    const button=document.getElementById('observation-forecast-toggle');
    if(!button)return;
    button.textContent=forecastView?'Show observation':'Preview next day';
    const dates=new Set((observationCatalog?.observations||[]).map(item=>item.date));
    const available=Boolean(currentObservedDate&&dates.has(previousObservationDate(currentObservedDate)));
    button.disabled=!forecastView&&!available;
    button.title=forecastView?'Return to the satellite observation.':available?'Uses two consecutive, checksummed observations; research-only estimate.':'A preceding consecutive satellite observation is not available for this date.';
  }
  function parseRouteWaypoints(raw){
    const lines=raw.split(/\r?\n/).map(line=>line.trim()).filter(Boolean);
    if(lines.length<2)return{error:'Add at least two waypoint lines to form a route.'};
    if(lines.length>500)return{error:'This screen accepts up to 500 waypoints per route.'};
    const coordinates=[];
    for(let index=0;index<lines.length;index++){
      const parts=lines[index].split(',').map(part=>part.trim());
      if(parts.length!==2||parts.some(part=>!part))return{error:`Waypoint ${index+1} must be written as longitude, latitude.`};
      const point=parts.map(Number);
      if(point.some(value=>!Number.isFinite(value)))return{error:`Waypoint ${index+1} contains a value that is not a number.`};
      if(point[0]<-180||point[0]>180||point[1]<-90||point[1]>90)return{error:`Waypoint ${index+1} must use longitude -180 to 180 and latitude -90 to 90.`};
      if(coordinates.length&&point[0]===coordinates.at(-1)[0]&&point[1]===coordinates.at(-1)[1])return{error:`Waypoint ${index+1} repeats the previous point.`};
      coordinates.push(point);
    }
    return{coordinates};
  }
  function renderRouteExposure(report){
    const results=document.getElementById('route-screen-result');
    const coverage=report.sampling||{};
    const centerline=report.sic_summary_fraction||{};
    const summary=report.context_band_sic_summary_fraction||centerline;
    document.getElementById('route-screen-result-title').textContent=`${document.getElementById('route-screen-name').value.trim()||'Route'} · observed context`;
    document.getElementById('route-result-length').textContent=`${Number(report.route_length_km).toFixed(1)} km`;
    const bandCount=coverage.context_band_valid_sample_count??coverage.valid_sample_count;
    const bandTotal=coverage.context_band_total_sample_count??coverage.total_sample_count;
    const bandCoverage=coverage.context_band_valid_sample_fraction??coverage.valid_sample_fraction;
    document.getElementById('route-result-coverage').textContent=`${bandCount} of ${bandTotal} · ${(bandCoverage*100).toFixed(1)}%`;
    document.getElementById('route-result-mean').textContent=Number.isFinite(summary.mean)?`${(summary.mean*100).toFixed(1)}%`:'No valid pixels';
    document.getElementById('route-result-p95').textContent=Number.isFinite(summary.p95)?`${(summary.p95*100).toFixed(1)}%`:'No valid pixels';
    document.getElementById('route-result-line-mean').textContent=Number.isFinite(centerline.mean)?`${(centerline.mean*100).toFixed(1)}%`:'No valid pixels';
    const provenance=report.source_provenance||{};
    const halfWidth=Number(coverage.context_band_half_width_km??0);
    const bandLabel=halfWidth>0?`${halfWidth} km on either side`:'line only';
    document.getElementById('route-result-provenance').textContent=`${provenance.product||provenance.dataset_id||'Satellite observation'} · ${provenance.source_file||'source file unavailable'} · observed ${report.observation_timestamp||report.observation_date}. Context: ${bandLabel}. Centerline coverage: ${report.data_coverage_status||'unknown'}. Band coverage: ${coverage.context_band_data_coverage_status||report.data_coverage_status||'unknown'}.`;
    const gate=report.evidence_gate||{};
    document.getElementById('route-evidence-status').textContent=gate.observation_display_status==='insufficient_coverage'?'Insufficient observation coverage · decision withheld':'Observed context only · decision withheld';
    const ageDays=Number.isFinite(gate.observation_age_hours_at_request)?Math.floor(gate.observation_age_hours_at_request/24):null;
    document.getElementById('route-evidence-reason').textContent=`${ageDays==null?'Dated observation':`Archive · ${ageDays} days old`} · ${(Number(gate.centerline_coverage_fraction||0)*100).toFixed(0)}% line coverage${gate.context_band_coverage_fraction==null?'':` · ${(Number(gate.context_band_coverage_fraction)*100).toFixed(0)}% band coverage`}. No calibrated forecast or safety assessment.`;
    const fragility=report.ice_edge_fragility||{};
    document.getElementById('route-fragility-summary').textContent=`${fragility.fragile_segment_count??0} of ${(fragility.segments||[]).length} sections change 15% ice-edge class with a hypothetical ±5-point ice change. ${fragility.unassessable_segment_count??0} lack enough data.`;
    const vessel=report.vessel_research_screen;
    const enteredLimit=routeReviewInput?.research_vessel_profile?.max_observed_sic_fraction;
    document.getElementById('route-vessel-screen').textContent=vessel?`Research vessel check: observed maximum ${vessel.observed_max_sic_fraction==null?'unavailable':`${(vessel.observed_max_sic_fraction*100).toFixed(0)}%`}${Number.isFinite(enteredLimit)?` vs entered limit ${(enteredLimit*100).toFixed(0)}%`:''}. ${vessel.status==='within_entered_observation_limits_only'?'Within entered observed-ice limits only.':'Limit exceeded or evidence incomplete.'} Profile unverified; no clearance.`:'No vessel limits entered. Add a research profile to compare observed ice with your own limits.';
    const body=document.getElementById('route-profile-body');body.replaceChildren();
    for(const bin of report.profile_5km||[]){
      const row=document.createElement('tr');
      const values=[`${Number(bin.start_km).toFixed(0)}-${Number(bin.end_km).toFixed(1)} km`,Number.isFinite(bin.context_band_mean_sic_fraction)?`${(bin.context_band_mean_sic_fraction*100).toFixed(1)}%`:Number.isFinite(bin.mean_sic_fraction)?`${(bin.mean_sic_fraction*100).toFixed(1)}%`:'No valid pixels',Number.isFinite(bin.context_band_maximum_sic_fraction)?`${(bin.context_band_maximum_sic_fraction*100).toFixed(1)}%`:Number.isFinite(bin.maximum_sic_fraction)?`${(bin.maximum_sic_fraction*100).toFixed(1)}%`:'No valid pixels',`${(Number(bin.context_band_coverage_fraction??bin.data_coverage_fraction??0)*100).toFixed(1)}%`];
      for(const value of values){const cell=document.createElement('td');cell.textContent=value;row.appendChild(cell);}
      body.appendChild(row);
    }
    const overlay=document.getElementById('observation-route-overlay');
    const path=document.getElementById('observation-route-path');
    const bandPath=document.getElementById('observation-context-band');
    const shape=(report.grid||{}).shape||[];const width=Number(shape[1]),height=Number(shape[0]);
    const points=report.centerline_pixel_coordinates||[];
    if(width>0&&height>0&&points.length){
      overlay.setAttribute('viewBox',`0 0 ${width} ${height}`);
      let drawing=false;const segments=[];
      for(const point of points){if(!Array.isArray(point)||point.length!==2||!point.every(Number.isFinite)){drawing=false;continue;}segments.push(`${drawing?'L':'M'}${point[0].toFixed(2)} ${point[1].toFixed(2)}`);drawing=true;}
      path.setAttribute('d',segments.join(' '));
      const polygon=report.context_band_pixel_polygon||[];
      const bandData=polygon.length>3&&polygon.every(point=>Array.isArray(point)&&point.length===2&&point.every(Number.isFinite))?`M${polygon.map(point=>`${point[0].toFixed(2)} ${point[1].toFixed(2)}`).join(' L')} Z`:'';
      bandPath.setAttribute('d',bandData);
      setSvgHidden(overlay,!segments.length&&!bandData);
      document.getElementById('map-canvas').dataset.routeVisible=String(Boolean(segments.length));
      document.getElementById('map-canvas').dataset.contextBandVisible=String(Boolean(bandData));
      document.querySelector('.map-legend .legend-item:nth-of-type(2) span').textContent='Analyzed centerline';
    }
    document.querySelector('.route-secondary-review').open=true;
    results.hidden=false;
  }
  function renderRouteCandidates(report){
    const results=document.getElementById('route-candidate-results');
    const body=document.getElementById('route-candidate-body');body.replaceChildren();
    const labels={shortest_grid_distance:['Shortest grid distance','shortest'],distance_ice_tradeoff:['Distance / ice trade-off','tradeoff'],lower_observed_ice:['Lower observed SIC','low-ice']};
    for(const candidate of report.candidates||[]){
      const [label,swatch]=labels[candidate.candidate_id]||[candidate.candidate_id,'shortest'];
      const metrics=candidate.metrics||{};const row=document.createElement('tr');
      const name=document.createElement('td');const nameWrap=document.createElement('span');nameWrap.className='route-candidate-name';const line=document.createElement('i');line.className=`candidate-swatch ${swatch==='shortest'?'':swatch}`;const text=document.createElement('span');text.textContent=label;nameWrap.append(line,text);name.appendChild(nameWrap);
      const distance=document.createElement('td');distance.textContent=Number.isFinite(metrics.grid_distance_km)?`${Number(metrics.grid_distance_km).toFixed(1)} km`:'—';
      const sic=document.createElement('td');sic.textContent=Number.isFinite(metrics.distance_weighted_mean_sic_fraction)?`${(Number(metrics.distance_weighted_mean_sic_fraction)*100).toFixed(1)}%`:'—';
      const researchTime=document.createElement('td');const time=candidate.research_transit_sensitivity||{};researchTime.textContent=time.status==='hypothetical'?`${Number(time.fast_hours).toFixed(1)}–${Number(time.slow_hours).toFixed(1)} h`:'Withheld';researchTime.title=time.status==='hypothetical'?'Hypothetical ±20% speed sensitivity; not a vessel ETA.':time.reason||'Research time unavailable.';
      const explanation=document.createElement('td');explanation.textContent=`${candidate.explanation||''}${candidate.distinct_from?` Same grid path as ${labels[candidate.distinct_from]?.[0]||candidate.distinct_from}.`:''}`;
      [name,distance,sic,researchTime,explanation].forEach(cell=>row.appendChild(cell));body.appendChild(row);
    }
    const assumptions=report.research_time_assumptions||{};document.getElementById('route-time-method').textContent=`Research assumptions: ${assumptions.open_water_knots??'—'} kn open water · ${assumptions.ice_affected_knots??'—'} kn ice-affected · 80–120% speed sweep. Not a confidence interval. No fuel estimate or vessel safety assessment.`;
    const sensitivity=document.getElementById('route-candidate-sensitivity');
    const scenarios=report.fixed_path_counterfactuals?.scenarios||[];
    const baseline=scenarios.find(item=>item.sic_change_fraction===0);
    const lower=scenarios.find(item=>item.sic_change_fraction===-0.05);
    const higher=scenarios.find(item=>item.sic_change_fraction===0.05);
    sensitivity.hidden=!(baseline&&lower&&higher);
    if(!sensitivity.hidden){
      const pathName=item=>labels[item.best_fixed_candidate_id]?.[0]||item.best_fixed_candidate_id||'unavailable';
      document.getElementById('route-candidate-sensitivity-result').textContent=`Preferred fixed path by exploratory score: −5 points: ${pathName(lower)} · observed: ${pathName(baseline)} · +5 points: ${pathName(higher)}. ${report.fixed_path_counterfactuals.preferred_fixed_alternative_changes?'The preferred fixed path changes under this test.':'The preferred fixed path stays the same under this test.'}`;
    }
    const grid=report.grid||{};const width=Number(grid.shape?.[1]),height=Number(grid.shape?.[0]);
    const overlay=document.getElementById('route-candidate-overlay');overlay.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const ids={shortest_grid_distance:'candidate-path-shortest',distance_ice_tradeoff:'candidate-path-tradeoff',lower_observed_ice:'candidate-path-low-ice'};
    for(const candidate of report.candidates||[]){
      const path=document.getElementById(ids[candidate.candidate_id]);const points=candidate.grid_pixel_coordinates||[];
      path.setAttribute('d',points.map((point,index)=>`${index?'L':'M'}${Number(point[0]).toFixed(2)} ${Number(point[1]).toFixed(2)}`).join(' '));
    }
    const hasCandidates=width>0&&height>0&&(report.candidates||[]).length>0;
    setSvgHidden(overlay,!hasCandidates);
    const map=document.getElementById('map-canvas');map.dataset.candidatesVisible=String(hasCandidates);
    document.querySelector('.map-legend .legend-item:nth-of-type(2) span').textContent='Compared path objectives';
    const provenance=report.source_provenance||{};const endpoints=report.endpoints||{};
    document.getElementById('route-candidate-provenance').textContent=`${provenance.product||provenance.dataset_id||'Satellite observation'} · ${provenance.source_file||'source file unavailable'} · observed ${report.observation_timestamp||report.observation_date}. Endpoints use nearest pixel centers (${Number(endpoints.grid_center_origin?.snap_distance_km||0).toFixed(2)} km and ${Number(endpoints.grid_center_destination?.snap_distance_km||0).toFixed(2)} km snap).`;
    results.hidden=false;
  }
  function readRoutePoint(prefix,label){
    const lonText=document.getElementById(`route-${prefix}-lon`).value.trim();const latText=document.getElementById(`route-${prefix}-lat`).value.trim();
    const longitude=Number(lonText),latitude=Number(latText);
    if(!lonText||!latText||!Number.isFinite(longitude)||!Number.isFinite(latitude)||longitude < -180||longitude > 180||latitude < -90||latitude > 90)return{error:`Enter a valid ${label} longitude and latitude in WGS84 decimal degrees.`};
    return{coordinates:[longitude,latitude]};
  }
  async function submitRouteCandidates(event){
    event.preventDefault();
    const errorNode=document.getElementById('route-candidate-error');errorNode.hidden=true;errorNode.textContent='';
    const origin=readRoutePoint('origin','origin');const destination=readRoutePoint('destination','destination');
    if(origin.error||destination.error){errorNode.textContent=origin.error||destination.error;errorNode.hidden=false;document.getElementById(origin.error?'route-origin-lon':'route-destination-lon').focus();return;}
    if(origin.coordinates[0]===destination.coordinates[0]&&origin.coordinates[1]===destination.coordinates[1]){errorNode.textContent='Choose different origin and destination points.';errorNode.hidden=false;return;}
    if(!observedView||!currentObservedDate){errorNode.textContent='Load a dated satellite observation before comparing paths.';errorNode.hidden=false;return;}
    const openSpeed=Number(document.getElementById('research-open-speed').value);const iceSpeed=Number(document.getElementById('research-ice-speed').value);
    if(!Number.isFinite(openSpeed)||openSpeed<=0||openSpeed>25||!Number.isFinite(iceSpeed)||iceSpeed<=0||iceSpeed>12||iceSpeed>openSpeed){errorNode.textContent='Enter positive assumed speeds within the shown limits; ice-affected speed must not exceed open-water speed.';errorNode.hidden=false;document.querySelector('.route-time-assumptions').open=true;document.getElementById('research-open-speed').focus();return;}
    clearRouteExposure();
    const requestId=routeCandidateRequestId;const controller=new AbortController();routeCandidateController=controller;
    const button=document.getElementById('route-candidate-submit');button.disabled=true;button.textContent='Comparing…';button.setAttribute('aria-busy','true');
    try{
      const report=await getJson(`${apiRoot()}/api/v1/route/candidates`,{method:'POST',timeoutMs:30000,signal:controller.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify({route_id:'Observed route comparison',observation_date:currentObservedDate,origin:{type:'Point',coordinates:origin.coordinates},destination:{type:'Point',coordinates:destination.coordinates},research_open_water_knots:openSpeed,research_ice_affected_knots:iceSpeed})});
      if(requestId!==routeCandidateRequestId)return;
      renderRouteCandidates(report);
    }catch(error){
      if(requestId!==routeCandidateRequestId)return;
      errorNode.textContent=error.name==='AbortError'?'The path comparison timed out. Check the service and retry.':error.status===401||error.status===403?'The service did not authorize this request. Sign in through the approved ministry access gateway, then retry.':error.status===404?'No observation is available for the selected date. Choose another date and retry.':error.status===422?`${error.message} Choose endpoints on valid pixels in the displayed scene and retry.`:error.status===413?'The observed grid is too large for this bounded path comparison.':error instanceof TypeError?'The route service could not be reached. Check the API connection and retry.':'The path comparison could not be completed. Retry later or contact the service administrator.';
      errorNode.hidden=false;
    }finally{if(requestId===routeCandidateRequestId){button.disabled=false;button.textContent='Compare observed paths';button.removeAttribute('aria-busy');routeCandidateController=null;}}
  }
  async function submitRouteExposure(event){
    event.preventDefault();
    const errorNode=document.getElementById('route-screen-error');errorNode.hidden=true;errorNode.textContent='';
    const parsed=parseRouteWaypoints(document.getElementById('route-screen-coordinates').value);
    if(parsed.error){errorNode.textContent=parsed.error;errorNode.hidden=false;document.getElementById('route-screen-coordinates').focus();return;}
    const profileLabel=document.getElementById('research-profile-label').value.trim();
    const maxIceText=document.getElementById('research-max-sic').value.trim();
    const minCoverageText=document.getElementById('research-min-coverage').value.trim();
    const hasProfile=Boolean(profileLabel||maxIceText||minCoverageText);
    const maxIce=Number(maxIceText),minCoverage=Number(minCoverageText);
    if(hasProfile&&(!profileLabel||!maxIceText||!minCoverageText||!Number.isFinite(maxIce)||!Number.isFinite(minCoverage)||maxIce<0||maxIce>100||minCoverage<0||minCoverage>100)){
      errorNode.textContent='Complete all three research vessel limits with percentages from 0 to 100, or leave them all empty.';errorNode.hidden=false;document.querySelector('.route-research-profile').open=true;return;
    }
    if(!observedView||!document.getElementById('observation-date').value){errorNode.textContent='Load a satellite observation before reviewing route exposure.';errorNode.hidden=false;return;}
    clearRouteExposure(true);
    const requestId=routeExposureRequestId;const controller=new AbortController();routeExposureController=controller;
    const button=document.getElementById('route-screen-submit');button.disabled=true;button.textContent='Reviewing…';button.setAttribute('aria-busy','true');
    try{
      const reviewInput={route_id:document.getElementById('route-screen-name').value.trim()||'Route review',observation_date:document.getElementById('observation-date').value,analysis_half_width_km:Number(document.getElementById('route-analysis-half-width').value),geometry:{type:'LineString',coordinates:parsed.coordinates}};
      if(hasProfile)reviewInput.research_vessel_profile={profile_label:profileLabel,max_observed_sic_fraction:maxIce/100,minimum_observation_coverage_fraction:minCoverage/100};
      const report=await getJson(`${apiRoot()}/api/v1/route/exposure`,{method:'POST',timeoutMs:20000,signal:controller.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify(reviewInput)});
      if(requestId!==routeExposureRequestId)return;
      routeReviewInput=reviewInput;
      routeReviewSourceHash=report?.source_provenance?.source_sha256||null;
      renderRouteExposure(report);
      window.dispatchEvent(new CustomEvent('southernpassage:route-reviewed',{detail:{request:reviewInput,sourceHash:routeReviewSourceHash}}));
    }catch(error){
      if(requestId!==routeExposureRequestId)return;
      const message=error.name==='AbortError'?'The route review timed out. Check the service and retry; your waypoints are still here.':error.status===401||error.status===403?'The service did not authorize this request. Sign in through the approved ministry access gateway, then retry.':error.status===404?'No observation is available for the selected date. Choose another date and retry.':error.status===422?error.message:error.status===413?'This route and context band exceed the analysis limit. Shorten the line or choose a narrower band, then retry.':error instanceof TypeError?'The route service could not be reached. Check the API connection and retry.':'The route review could not be completed. Retry later or contact the service administrator.';
      errorNode.textContent=message;errorNode.hidden=false;
    }finally{
      if(requestId===routeExposureRequestId){button.disabled=false;button.textContent='Review observed exposure';button.removeAttribute('aria-busy');routeExposureController=null;}
    }
  }
  async function getRouteReviewBundle(){
    if(routeReviewBundle)return routeReviewBundle;
    if(!routeReviewInput)throw new Error('Review the route first.');
    if(!routeReviewBundlePromise){
      const input=routeReviewInput;
      routeReviewBundlePromise=getJson(`${apiRoot()}/api/v1/reviews/route/bundle`,{method:'POST',timeoutMs:20000,headers:{'Content-Type':'application/json'},body:JSON.stringify(input)}).then(bundle=>{
        const packet=bundle?.packet,feature=bundle?.geojson?.features?.[0];
        if(routeReviewInput!==input||packet?.review?.status!=='awaiting_human_review'||!packet?.integrity?.sha256||packet?.observation?.source_provenance?.source_sha256!==routeReviewSourceHash||feature?.properties?.packet_sha256!==packet.integrity.sha256||feature?.properties?.packet_id!==packet.packet_id)throw new Error('The route or source changed. Review the route again.');
        routeReviewBundle=bundle;return bundle;
      }).finally(()=>{routeReviewBundlePromise=null;});
    }
    return routeReviewBundlePromise;
  }
  async function downloadRouteReviewPacket(){
    if(!routeReviewInput)return;
    const button=document.getElementById('route-export-button');
    const status=document.getElementById('route-export-status');
    const controller=new AbortController();routeExportController=controller;
    const requestInput=routeReviewInput;
    button.disabled=true;button.textContent='Preparing packet…';button.setAttribute('aria-busy','true');
    status.textContent='Verifying the selected observation and preparing the review record.';
    try{
      const {packet}=await getRouteReviewBundle();
      if(controller.signal.aborted||packet?.review?.status!=='awaiting_human_review'||!packet?.integrity?.sha256)throw new Error('The review packet was incomplete.');
      if(!routeReviewSourceHash||packet?.observation?.source_provenance?.source_sha256!==routeReviewSourceHash){status.textContent='The source observation changed. Review the route again before downloading.';return;}
      const filename=`southern-passage-review-${String(requestInput.route_id).replace(/[^a-z0-9_-]+/gi,'-').slice(0,48)||'route'}-${requestInput.observation_date}.json`;
      const url=URL.createObjectURL(new Blob([JSON.stringify(packet,null,2)+'\n'],{type:'application/json'}));
      const link=document.createElement('a');link.href=url;link.download=filename;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
      status.textContent='Review packet downloaded. A human decision is still required.';
    }catch(error){
      if(controller.signal.aborted)return;
      status.textContent=error.status===401||error.status===403?'Access was denied. Sign in through the portal and retry.':error.status===404||error.status===422?'The selected observation or route is no longer valid. Review the route again.':error instanceof TypeError?'The packet service could not be reached. Check the connection and retry.':'The packet could not be prepared. Retry or contact the service administrator.';
    }finally{
      if(routeExportController===controller){routeExportController=null;button.disabled=false;button.textContent='Download review packet';button.removeAttribute('aria-busy');}
    }
  }
  async function downloadRouteGeoJSON(){
    if(!routeReviewInput)return;
    const button=document.getElementById('route-geojson-button');
    const status=document.getElementById('route-export-status');
    button.disabled=true;button.textContent='Preparing map data…';button.setAttribute('aria-busy','true');
    try{
      const {geojson:geo}=await getRouteReviewBundle();
      const feature=geo?.features?.[0];
      if(geo?.type!=='FeatureCollection'||feature?.properties?.source_sha256!==routeReviewSourceHash){status.textContent='The source observation changed. Review the route again before downloading.';return;}
      const filename=`southern-passage-${String(routeReviewInput.route_id).replace(/[^a-z0-9_-]+/gi,'-').slice(0,48)||'route'}-${routeReviewInput.observation_date}.geojson`;
      const url=URL.createObjectURL(new Blob([JSON.stringify(geo,null,2)+'\n'],{type:'application/geo+json'}));
      const link=document.createElement('a');link.href=url;link.download=filename;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
      status.textContent='Map data downloaded. It is an observed-data review export, not a cleared route.';
    }catch(error){
      status.textContent=error.status===401||error.status===403?'Access was denied. Sign in through the portal and retry.':'Map data could not be prepared. Review the route again or retry later.';
    }finally{button.disabled=false;button.textContent='Download map data';button.removeAttribute('aria-busy');}
  }
  async function loadObservationCatalog(){
    const button=document.getElementById('observations-button');
    const error=document.getElementById('observation-load-error');
    button.setAttribute('aria-busy','true');
    button.title='Verifying the dated, checksummed NOAA VIIRS catalog.';
    error.dataset.state='loading';
    error.textContent='Verifying observation dates and source checksums…';
    error.hidden=false;
    try{
      const [catalog,sourceStatus]=await Promise.all([
        getJson(`${apiRoot()}/api/v1/observations/sic`,{timeoutMs:30000}),
        getJson(`${apiRoot()}/api/v1/observations/sic/source-status`,{timeoutMs:10000})
      ]);
      observationCatalog={...catalog,sourceStatus};
      if(!Array.isArray(observationCatalog.observations))throw new Error('Observation dates are missing from the data catalog.');
      const select=document.getElementById('observation-date');
      const datedObservations=observationCatalog.observations.filter(item=>/^\d{4}-\d{2}-\d{2}$/.test(item.date));
      datedObservations.forEach(item=>{const option=document.createElement('option');option.value=item.date;option.textContent=item.date;select.appendChild(option);});
      if(!currentObservedDate&&datedObservations.length)select.value=datedObservations.at(-1).date;
      button.disabled=!observationCatalog.observations.length;
      document.getElementById('replay-observation-button').disabled=button.disabled;
      const observationsNav=document.getElementById('nav-observations');
      observationsNav.disabled=button.disabled;observationsNav.setAttribute('aria-disabled',String(button.disabled));
      observationsNav.title=button.disabled?'No dated satellite observations are available.':'Open dated satellite observations';
      button.title=button.disabled?'No dated VIIRS observations are available.':'Show the actual NOAA VIIRS observation grid.';
      error.hidden=true;
      if(button.disabled){error.textContent='The observation service is connected but returned no usable dates. The scenario view remains available.';error.hidden=false;}
    }catch(errorResponse){button.disabled=true;document.getElementById('replay-observation-button').disabled=true;const observationsNav=document.getElementById('nav-observations');observationsNav.disabled=true;observationsNav.setAttribute('aria-disabled','true');observationsNav.title='Satellite observations are unavailable';button.title='NOAA VIIRS observations are unavailable; start the local API with the dataset mounted.';error.dataset.state='error';error.textContent='NOAA VIIRS observations could not be loaded. Start the API with the observation dataset mounted, then reload this page.';error.hidden=false;}
    finally{button.removeAttribute('aria-busy');}
  }
  function colorBytes(hex){const value=parseInt(hex.slice(1),16);return[(value>>16)&255,(value>>8)&255,value&255];}
  function drawObservation(data){
    const canvas=document.getElementById('observation-map');
    const rows=data.sic_fraction, height=rows.length, width=rows[0]?.length||0;
    if(!width||!height||height*width>100000)throw new Error('Observation grid has an unsupported size.');
    canvas.width=width;canvas.height=height;
    const context=canvas.getContext('2d',{alpha:true});
    const pixels=context.createImageData(width,height);
    let minX=width,minY=height,maxX=-1,maxY=-1;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const index=(y*width+x)*4, value=rows[y][x];
      if(value===null||!Number.isFinite(value)){pixels.data[index+3]=0;continue;}
      minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);
      const [r,g,b]=colorBytes(fieldColor(clamp(value*100,0,100)));
      pixels.data[index]=r;pixels.data[index+1]=g;pixels.data[index+2]=b;pixels.data[index+3]=255;
    }
    context.putImageData(pixels,0,0);
    const map=document.getElementById('map-canvas');
    if(maxX>=minX&&maxY>=minY){
      const validWidth=maxX-minX+1,validHeight=maxY-minY+1;
      const zoom=Math.max(1,Math.min(1.7,.9*width/validWidth,.9*height/validHeight));
      const panX=-((minX+maxX+1)/2-width/2)/width*zoom*100;
      const panY=-((minY+maxY+1)/2-height/2)/height*zoom*100;
      map.style.setProperty('--observation-zoom',zoom.toFixed(3));
      map.style.setProperty('--observation-pan-x',`${panX.toFixed(2)}%`);
      map.style.setProperty('--observation-pan-y',`${panY.toFixed(2)}%`);
    }else{
      map.style.setProperty('--observation-zoom','1');
      map.style.setProperty('--observation-pan-x','0%');
      map.style.setProperty('--observation-pan-y','0%');
    }
  }
  function formatCoordinate(value,positive,negative){return `${Math.abs(value).toFixed(1)}°${value<0?negative:positive}`;}
  function transitionWorkspace(){
    const shell=document.querySelector('.app-shell');
    shell.classList.remove('view-arrived');
    void shell.offsetWidth;
    shell.classList.add('view-arrived');
    window.setTimeout(()=>shell.classList.remove('view-arrived'),440);
  }
  function setWorkArea(area){
    const shell=document.querySelector('.app-shell');
    shell.dataset.workArea=area;
    document.querySelectorAll('[data-workspace-target]').forEach(button=>{
      const active=button.dataset.workspaceTarget===area;
      button.classList.toggle('is-active',active);
      if(active)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');
    });
  }
  async function showObservation(day){
    stopReplay();
    const requestId=++observationRequestId;
    researchForecastRequestId++;researchForecastController?.abort();researchForecastController=null;forecastView=false;
    clearRouteExposure();
    const note=document.getElementById('observation-time-note');
    const select=document.getElementById('observation-date');
    document.getElementById('route-screen').hidden=true;select.disabled=true;
    select.setAttribute('aria-busy','true');note.textContent='Loading the observed SIC grid…';
    try{
      const data=await getJson(`${apiRoot()}/api/v1/observations/sic/${encodeURIComponent(day)}`);
      if(requestId!==observationRequestId)return;
      currentObservationData=data;
      drawObservation(data);
      const canvas=document.getElementById('observation-map');
      const map=document.getElementById('map-canvas');
      const shell=document.querySelector('.app-shell');
      const center=observationCatalog?.selection?.center_lon_lat;
      const location=center?` · ${formatCoordinate(center[1],'N','S')}, ${formatCoordinate(center[0],'E','W')}`:'';
      document.getElementById('map-title-text').textContent='Observed sea ice';
      const platform=data.source_provenance.platform;
      const sensorLabel=platform==='NOAA-21'?'NOAA-21':platform==='NOAA-20'?'NOAA-20':platform==='S-NPP'?'S-NPP':'NOAA';
      document.getElementById('map-subtitle-text').textContent=`${sensorLabel} VIIRS${location}`;
      document.getElementById('map-stamp-label').textContent=`OBSERVED · ${data.date}`;
      document.getElementById('page-title').firstChild.textContent='Observed sea ice';
      document.getElementById('page-title-accent').textContent='';
      document.getElementById('page-intro').textContent='Dated satellite concentration for evidence review. No route clearance.';
      document.getElementById('replay-evidence-title').textContent='Ice motion evidence';
      document.getElementById('observation-colorbar-label').textContent='Observed ice';
      document.querySelector('.demo-label').textContent='SATELLITE SNAPSHOT';
      document.querySelector('.map-legend .legend-head').firstChild.textContent='VIIRS SIC ';
      document.querySelector('.map-legend .legend-item span').textContent='Sea-ice concentration';
      document.querySelector('.projection-label').lastChild.textContent=' VIIRS GRID';
      canvas.setAttribute('aria-label',`Observed NOAA ${sensorLabel} VIIRS sea-ice concentration on ${data.date}; ${data.grid.crs}, ${data.grid.pixel_spacing_x_m.toFixed(0)} metre pixels`);
      setSvgHidden(document.getElementById('polar-map'),true);canvas.hidden=false;
      document.getElementById('observation-controls').hidden=false;
      document.getElementById('route-screen').hidden=false;
      currentObservedDate=data.date;
      map.dataset.view='observed';shell.dataset.mapView='observed';observedView=true;
      setWorkArea('observations');
      transitionWorkspace();
      updateForecastToggle();
      const button=document.getElementById('observations-button');button.textContent='SHOW SCENARIO';button.setAttribute('aria-pressed','true');button.title='Return to the illustrative scenario map.';
      const newest=observationCatalog?.freshness;
      const ageHours=(Date.now()-Date.parse(data.timestamp))/3600000;
      const ageLabel=!Number.isFinite(ageHours)?'age unknown':ageHours<0?'future timestamp':ageHours<48?`${Math.round(ageHours)} hours old`:`${Math.floor(ageHours/24)} days old`;
      const sourceStatus=observationCatalog?.sourceStatus;
      const sourceProducts=sourceStatus?.products||[];
      const latestSource=sourceProducts.map(item=>item.latest_source_timestamp).filter(Boolean).sort().at(-1);
      const sourceLabel=sourceStatus?.available
        ?`NOAA source metadata through ${latestSource||'unknown'} (${sourceStatus.source_status}; monitor ${sourceStatus.monitor_status})`
        :'NOAA source metadata is not being monitored';
      note.dataset.freshness=newest?.status||'unavailable';
      note.title=`${newest?.note||'Observation age could not be determined.'} ${sourceStatus?.warning||''}`;
      note.textContent=`Archive · ${data.date} · ${ageLabel}`;
      document.getElementById('observation-provenance-text').textContent=`Observed ${data.timestamp}. Mounted catalog newest: ${newest?.newest_timestamp||'unknown'} (${newest?.status||'unavailable'}). ${sourceLabel}. Sparse satellite sampling; not a navigation recommendation.`;
    }catch(error){
      if(requestId!==observationRequestId)return;
      if(currentObservedDate)select.value=currentObservedDate;
      document.getElementById('route-screen').hidden=!observedView;
      note.textContent=error.name==='AbortError'?'The observation request timed out. Try selecting this date again.':error.status===404?'No satellite observation exists for this date. Choose another listed date.':error.status===503?'Observation data are not available on the service. Return to the illustrative scenario or contact the data administrator.':error instanceof TypeError?'Observation service could not be reached. Check the API connection and try again.':'Could not load this observation. Try again or choose another listed date.';
    }finally{if(requestId===observationRequestId){select.disabled=false;select.removeAttribute('aria-busy');}}
  }
  async function showResearchForecast(){
    if(forecastView&&currentObservedDate){showObservation(currentObservedDate);return;}
    if(!currentObservedDate||!currentObservationData)return;
    clearRouteExposure();
    researchForecastRequestId++;researchForecastController?.abort();
    const requestId=researchForecastRequestId;const controller=new AbortController();researchForecastController=controller;
    const button=document.getElementById('observation-forecast-toggle');
    const note=document.getElementById('observation-time-note');
    button.disabled=true;button.textContent='Preparing…';button.setAttribute('aria-busy','true');
    note.textContent='Building a one-day research estimate from two consecutive, checksummed VIIRS observations.';
    try{
      const report=await getJson(`${apiRoot()}/api/v1/observations/sic/${encodeURIComponent(currentObservedDate)}/forecast`,{timeoutMs:30000,signal:controller.signal});
      if(requestId!==researchForecastRequestId)return;
      if(!Array.isArray(report.sic_fraction)||!report.target_date||report.decision_status!=='research_only')throw new Error('The research outlook response is incomplete.');
      drawObservation(report);
      forecastView=true;
      const inputs=report.source_provenance?.input_observations||[];
      document.getElementById('map-title-text').textContent='Research ice estimate';
      document.getElementById('map-subtitle-text').textContent=`Modelled from VIIRS observations ${inputs.map(item=>item.date).join(' and ')}`;
      document.getElementById('map-stamp-label').textContent=`ESTIMATED · ${report.target_date}`;
      document.getElementById('page-title').firstChild.textContent='Research ice estimate';
      document.getElementById('page-title-accent').textContent='';
      document.getElementById('page-intro').textContent='Experimental one-day concentration estimate. No calibrated route or safety assessment.';
      document.getElementById('replay-evidence-title').textContent='Ice motion evidence';
      document.getElementById('observation-colorbar-label').textContent='Estimated ice';
      document.querySelector('.demo-label').textContent='RESEARCH ONLY';
      document.querySelector('.map-legend .legend-head').firstChild.textContent='ESTIMATED SIC ';
      document.querySelector('.map-legend .legend-item span').textContent='Estimated concentration';
      document.getElementById('observation-map').setAttribute('aria-label',`Research-only one-day sea-ice concentration estimate for ${report.target_date}, initialized from observations ${inputs.map(item=>item.date).join(' and ')}`);
      document.getElementById('route-screen').hidden=true;
      const transfer=report.model_evidence?.regional_cross_product_transfer_check;
      const transferText=transfer&&Number.isFinite(Number(transfer.rmse_sic_fraction))&&Number.isFinite(Number(transfer.persistence_rmse_sic_fraction))
        ?` Earlier regional VIIRS transfer check: RMSE ${Number(transfer.rmse_sic_fraction).toFixed(5)} vs persistence ${Number(transfer.persistence_rmse_sic_fraction).toFixed(5)}; sparse and not forecast-specific.`:'';
      const coverage=Number(report.input_coverage?.common_coverage_fraction);
      note.dataset.freshness='research_only';
      note.title=report.warning||'Research-only model output; not an operational forecast.';
      note.textContent=`Experimental SIC estimate for ${report.target_date}. Inputs: ${inputs.map(item=>item.date).join(', ')}; ${(coverage*100).toFixed(1)}% common valid coverage (research display floor). No calibrated uncertainty; no route or safety assessment.${transferText}`;
      document.getElementById('observation-provenance-text').textContent=note.textContent;
      note.textContent=`Research estimate · ${report.target_date} · not validated for routes`;
      updateForecastToggle();
    }catch(error){
      if(requestId!==researchForecastRequestId)return;
      const prefix=error.status===422?'A one-day estimate is unavailable for this date: ':'';
      note.textContent=error.name==='AbortError'?'The estimate request timed out. The observed grid is unchanged.':error.status===401||error.status===403?'The service did not authorize this estimate. Sign in through the approved gateway, then retry.':error.status===404||error.status===422?`${prefix}${error.message}`:error.status===503?`The research model or observations are unavailable: ${error.message}`:error instanceof TypeError?'The estimate service could not be reached. The observed grid is unchanged.':'Could not create the research estimate. The observed grid is unchanged.';
      note.dataset.freshness='unavailable';
    }finally{
      if(requestId===researchForecastRequestId){button.removeAttribute('aria-busy');researchForecastController=null;updateForecastToggle();}
    }
  }
  function showScenario(){
    observationRequestId++;clearRouteExposure();
    researchForecastRequestId++;researchForecastController?.abort();researchForecastController=null;forecastView=false;currentObservationData=null;
    currentObservedDate=null;
    const button=document.getElementById('observations-button');
    document.querySelector('.app-shell').removeAttribute('data-map-view');
    document.getElementById('map-canvas').removeAttribute('data-view');
    document.getElementById('observation-map').hidden=true;setSvgHidden(document.getElementById('polar-map'),false);
    document.getElementById('observation-controls').hidden=true;
    updateForecastToggle();
    document.getElementById('route-screen').hidden=true;
    document.getElementById('map-title-text').textContent='Route scenario';
    document.getElementById('map-subtitle-text').textContent='Antarctic Peninsula';
    document.getElementById('map-stamp-label').textContent='SIMULATED ROUTE';
    document.getElementById('page-title').firstChild.textContent='Scenario routes';
    document.getElementById('page-title-accent').textContent='';
    document.getElementById('page-intro').textContent='Explore illustrative routes and a simulated vessel replay. No live tracking or route clearance.';
    document.getElementById('replay-evidence-title').textContent='Ice motion evidence';
    document.querySelector('.demo-label').textContent='SCENARIO WORKSPACE';
    document.querySelector('.map-legend .legend-head').firstChild.textContent='MAP KEY ';
    document.querySelector('.map-legend .legend-item span').textContent='Ice concentration';
    document.querySelector('.projection-label').lastChild.textContent=' POLAR VIEW';
    button.textContent='OBSERVED DATA';button.setAttribute('aria-pressed','false');button.title='Show the actual NOAA VIIRS observation grid.';
    document.getElementById('observation-time-note').textContent='Dated satellite archive';
    observedView=false;
    setWorkArea('scenario');
    transitionWorkspace();
  }
  async function loadTransectEvidence(){
    const message=document.getElementById('transect-evidence-message');
    try{
      const report=await getJson(`${apiRoot()}/api/v1/research/transect-evidence`,{timeoutMs:8000});
      if(report.scope!=='frozen_transect_prospective_research_evidence'||report.navigation_clearance!==false||!Array.isArray(report.leads))throw new Error('Incomplete evidence');
      const count=Number(report.verified_case_count);
      const best=report.leads.find(item=>Number(item.evaluable_bulletins)>0);
      const rejected=Number(report.rejected_reports);
      if(!Number.isInteger(count)||count<0||!Number.isInteger(rejected)||rejected<0)throw new Error('Invalid evidence counts');
      if(rejected){message.textContent=`Prospective research evidence needs review: ${rejected} report${rejected===1?'':'s'} failed integrity checks. No route decision or navigation clearance.`;return;}
      if(!count){message.textContent=`Prospective route-line validation is collecting cases from ${report.eligible_bulletins_from}. No forecast–observation matches yet; uncertainty remains uncalibrated.`;return;}
      const leadSummary=best?` First evaluable lead: +${best.lead_days} days, ${best.evaluable_bulletins} independent bulletin${best.evaluable_bulletins===1?'':'s'}.`:'';
      message.textContent=`${count} sample-checked research case${count===1?'':'s'}.${leadSummary} Date-blocked uncertainty and vessel-route performance remain unapproved. No navigation clearance.`;
    }catch(error){
      message.textContent='Prospective forecast evidence is unavailable. Check the local API or verification archive; no route decision is inferred.';
    }
  }
  async function showResearchModelStatus(){
    const panel=document.querySelector('.research-status');
    const message=document.getElementById('research-status-message');
    const root=apiRoot();
    const controller=new AbortController();
    const timeout=setTimeout(()=>controller.abort(),3500);
    let recentTransferSummary='';
    let rollingClimatologySummary='';
    try{
      const response=await fetch(`${root}/api/v1/model`,{credentials:'include',signal:controller.signal,headers:{Accept:'application/json'}});
      if(!response.ok)throw new Error(`API returned ${response.status}`);
      const data=await response.json();
      if(!data||!data.model_id)throw new Error('Model metadata is incomplete');
      panel.dataset.state='connected';
      const recentTransfers=[['S-NPP',data.snpp_2026_transfer],['NOAA-21',data.noaa21_2026_transfer],['NOAA-20',data.noaa20_2026_transfer]]
        .filter(([,report])=>report?.available&&report.metrics)
        .map(([platform,report])=>{
          const rmse=Number(report.metrics.rmse),baseline=Number(report.metrics.persistence_rmse);
          const samples=Number(report.sample_scope?.valid_grid_cell_samples);
          const targetDays=Number(report.sample_scope?.target_days);
          const period=report.period?`${report.period.first_target_date} to ${report.period.last_target_date}`:'the reported period';
          return Number.isFinite(rmse)&&Number.isFinite(baseline)
            ?`${platform} ${rmse.toFixed(5)} vs ${baseline.toFixed(5)} persistence (${Number.isFinite(targetDays)?`${targetDays} target days, `:''}${period}; ${samples.toLocaleString()} valid cell-times)`:'';
        }).filter(Boolean);
      recentTransferSummary=recentTransfers.length
        ?` Separate VIIRS transfer checks from one 100-km region: ${recentTransfers.join('; ')}. Small, correlated samples; not route-scale evidence.`:'';
      const holdout=data.same_product_holdout;
      if(holdout&&holdout.available&&holdout.metrics){
        const metrics=holdout.metrics;
        const weighted=holdout.area_weighted?.overall;
        const score=weighted||metrics;
        const rmse=Number(score.rmse),baseline=Number(score.persistence_rmse);
        if(Number.isFinite(rmse)&&Number.isFinite(baseline)){
          const period=holdout.period?`${holdout.period.first_target_date} to ${holdout.period.last_target_date}`:'2017-2024';
          const regional=holdout.area_weighted?.by_latitude_band?.north_of_60S;
          const northernLoss=regional&&Number(regional.rmse)>Number(regional.persistence_rmse);
          const weighting=weighted?'area-weighted':'equal grid-cell weighting';
          const edge=holdout.area_weighted?.ice_edge?.overall;
          const edgeRmse=Number(edge?.mean_daily_iiee_million_km2),edgePersistence=Number(edge?.persistence_mean_daily_iiee_million_km2);
          const edgeSummary=Number.isFinite(edgeRmse)&&Number.isFinite(edgePersistence)?` 15%-threshold ice-edge disagreement: ${edgeRmse.toFixed(3)} vs ${edgePersistence.toFixed(3)} million km²/day.`:'';
          const rolling=data.rolling_origin_benchmark;
          const rollingMetrics=rolling?.available?rolling.area_weighted?.overall:null;
          const rollingRmse=Number(rollingMetrics?.rmse),rollingPersistence=Number(rollingMetrics?.persistence_rmse);
          const rollingClimatology=Number(rolling?.area_weighted?.monthly_climatology?.rmse);
          if(Number.isFinite(rollingClimatology))rollingClimatologySummary=` Fold-specific monthly climatology (fit only on earlier years) RMSE: ${rollingClimatology.toFixed(5)}; compare with the ten-fold regression and persistence scores above.`;
          const sensorEra=data.sensor_era_transfer;
          const sensorMetrics=sensorEra?.available?sensorEra.area_weighted?.overall||sensorEra.metrics:null;
          const sensorRmse=Number(sensorMetrics?.rmse),sensorPersistence=Number(sensorMetrics?.persistence_rmse);
          const sensorPeriod=sensorEra?.period?`${sensorEra.period.first_target_date} to ${sensorEra.period.last_target_date}`:'2025';
          const sensorEdge=sensorEra?.area_weighted?.ice_edge?.overall;
          const sensorEdgeRmse=Number(sensorEdge?.mean_daily_iiee_million_km2),sensorEdgePersistence=Number(sensorEdge?.persistence_mean_daily_iiee_million_km2);
          const sensorEdgeSummary=Number.isFinite(sensorEdgeRmse)&&Number.isFinite(sensorEdgePersistence)?` 15%-threshold edge disagreement ${sensorEdgeRmse.toFixed(6)} vs ${sensorEdgePersistence.toFixed(6)} million km²/day (effectively tied).`:'';
          const sensorSummary=Number.isFinite(sensorRmse)&&Number.isFinite(sensorPersistence)?` Separate 2025 sensor-era check (${sensorPeriod}; AMSR2-era): SIC RMSE ${sensorRmse.toFixed(5)} vs ${sensorPersistence.toFixed(5)} persistence;${sensorEdgeSummary} do not pool or tune against this result.`:'';
          const rollingSummary=`${edgeSummary}${sensorSummary}${Number.isFinite(rollingRmse)&&Number.isFinite(rollingPersistence)?` Ten-fold hindcast (1997-2016; separate fit per fold): ${rollingRmse.toFixed(5)} vs ${rollingPersistence.toFixed(5)} persistence; not the packaged model's score.`:''}`;
          message.textContent=`Same-product holdout (${period}): ${weighting} SIC RMSE ${rmse.toFixed(5)} vs persistence ${baseline.toFixed(5)}. Small average gain; slightly worse April-August${northernLoss?' and north of 60°S':''}.${rollingSummary} Research benchmark only; the map remains illustrative.`;
          return;
        }
      }
      const evaluation=data.external_evaluation;
      if(evaluation&&evaluation.available&&evaluation.metrics){
        const metrics=evaluation.metrics;
        const rmse=Number(metrics.rmse),baseline=Number(metrics.persistence_rmse);
        if(Number.isFinite(rmse)&&Number.isFinite(baseline)){
          const comparison=rmse<baseline?'better than':'worse than';
          const period=evaluation.period?` (${evaluation.period.first_target_date} to ${evaluation.period.last_target_date})`:'';
          const completeness=evaluation.dataset_complete?'download complete':'download partial';
          const scope=evaluation.spatial_evaluation?.method?.includes('block mean')?'on approximate 25 km block means, not a formal reprojection':'';
          message.textContent=`Research API connected · independent check${period}: RMSE ${rmse.toFixed(3)} vs persistence ${baseline.toFixed(3)} (${comparison}); ${completeness}${scope?`; ${scope}`:''}. Sparse satellite observation dates; research diagnostic only. The map remains illustrative and does not use this model.`;
          return;
        }
      }
      message.textContent='Research API connected, but no usable validation report is available. The map remains illustrative and does not use this model.';
    }catch(error){
      panel.dataset.state='error';
      message.textContent=error.name==='AbortError'?'Research API did not respond in time. Start the local API to view model status; the map remains illustrative and does not use this model.':'Research API is not available. Start the local API to view model status; the map remains illustrative and does not use this model.';
    }finally{
      if(recentTransferSummary)message.append(document.createTextNode(recentTransferSummary));
      if(rollingClimatologySummary)message.append(document.createTextNode(rollingClimatologySummary));
      clearTimeout(timeout);
    }
  }
  async function loadMotionEvidence(){
    const panel=document.getElementById('motion-evidence');
    const value=document.getElementById('motion-evidence-value');
    const detail=document.getElementById('motion-evidence-detail');
    try{
      const report=await getJson(`${apiRoot()}/api/v1/research/ice-motion`,{timeoutMs:8000});
      if(!report.available||report.decision_status!=='research_only')throw new Error('Verified report unavailable');
      const mae=Number(report.mae?.reduction_percent),rmse=Number(report.rmse?.reduction_percent);
      if(!Number.isFinite(mae)||!Number.isFinite(rmse)||report.fit_year!==2023||report.evaluation_year!==2024)throw new Error('Result metadata invalid');
      panel.dataset.state='verified';
      value.textContent='Retrospective result verified';
      document.getElementById('motion-mae').textContent=`−${mae.toFixed(2)}%`;
      document.getElementById('motion-rmse').textContent=`−${rmse.toFixed(2)}%`;
      detail.textContent=`Coefficient fitted in 2023, frozen for ${report.evaluated_days} evaluable days in 2024; ${Number(report.correlated_cell_days).toLocaleString()} correlated cell-days. ${report.zero_coverage_days} zero-coverage dates excluded. Same-product retrospective evidence only—not the animated map or a live route forecast.`;
      const evidenceLink=document.getElementById('motion-evidence-link');evidenceLink.href=`${apiRoot()}/api/v1/research/ice-motion`;evidenceLink.hidden=false;
    }catch(error){
      panel.dataset.state='unavailable';
      value.textContent='Verified result unavailable';
      detail.textContent='Start the local research API with the checksum-pinned report mounted. The simulated voyage remains available, but no result is substituted.';
    }
  }
  function stopReplay(){
    state.playing=false;
    if(replayFrame!==null){cancelAnimationFrame(replayFrame);replayFrame=null;}
    const button=document.getElementById('play-button');
    button.innerHTML='<svg class="play-icon" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 2.2 9.5 6 3 9.8Z"/></svg>';
    button.setAttribute('aria-label','Play simulated vessel replay');
  }
  function startReplay(){
    if(observedView)return;
    state.playing=true;
    state.vesselProgress=0;state.day=0;
    document.getElementById('timeline').value='0';
    render();
    const button=document.getElementById('play-button');
    button.innerHTML='<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3.5 2.5h2v7h-2zm3 0h2v7h-2z"/></svg>';
    button.setAttribute('aria-label','Pause simulated vessel replay');
    const started=performance.now(),duration=12000;
    const tick=now=>{
      if(!state.playing)return;
      const progress=Math.min(7,(now-started)/duration*7),day=Math.floor(progress);
      state.vesselProgress=progress;
      document.getElementById('timeline').value=String(progress);
      if(day!==state.day){state.day=day;render();}else updateVesselPosition();
      if(progress<7)replayFrame=requestAnimationFrame(tick);else stopReplay();
    };
    replayFrame=requestAnimationFrame(tick);
  }
  document.querySelectorAll('.horizon').forEach(b=>b.addEventListener('click',()=>{state.horizon=+b.dataset.day;document.querySelectorAll('.horizon').forEach(x=>{const active=x===b;x.classList.toggle('active',active);x.setAttribute('aria-pressed',String(active));});render();}));
  document.querySelectorAll('.route-option').forEach(b=>b.addEventListener('click',()=>{state.route=b.dataset.route;document.querySelectorAll('.route-option').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));render();}));
  document.querySelectorAll('[data-endpoint]').forEach(button=>button.addEventListener('click',()=>{
    endpointPick=button.dataset.endpoint;
    document.querySelectorAll('[data-endpoint]').forEach(x=>x.setAttribute('aria-pressed',String(x===button)));
    const label=endpointPick==='origin'?'departure':'arrival';
    document.getElementById('endpoint-status').textContent=`Click open water on the map to set ${label}.`;
    svg.classList.add('picking-endpoint');
    svg.focus({preventScroll:true});
  }));
  document.getElementById('reset-endpoints').addEventListener('click',()=>{
    Object.assign(origin,defaultOrigin);Object.assign(destination,defaultDestination);
    originLabel='Rothera';destinationLabel='Halley VI';endpointPick=null;
    document.querySelectorAll('[data-endpoint]').forEach(x=>x.setAttribute('aria-pressed','false'));
    svg.classList.remove('picking-endpoint');render();
  });
  svg.addEventListener('click',event=>{
    if(!endpointPick||observedView)return;
    const matrix=svg.getScreenCTM();if(!matrix)return;
    const point=new DOMPoint(event.clientX,event.clientY).matrixTransform(matrix.inverse());
    const candidate={x:point.x,y:point.y};
    const cell=toCell(candidate),snapped=cellPoint(cell.c,cell.r),bergList=bergs();
    if(dist(candidate,{x:480,y:356})>294||isLand(candidate)||dist(snapped,{x:480,y:356})>294||isLand(snapped)||bergList.some(berg=>dist(snapped,berg)<berg.radius+10)){
      document.getElementById('endpoint-status').textContent='That point is land or too close to an iceberg. Choose open water.';return;
    }
    Object.assign(endpointPick==='origin'?origin:destination,snapped);
    if(endpointPick==='origin')originLabel='Start';else destinationLabel='Arrival';
    endpointPick=null;document.querySelectorAll('[data-endpoint]').forEach(x=>x.setAttribute('aria-pressed','false'));
    svg.classList.remove('picking-endpoint');render();
  });
  document.querySelectorAll('.layer-tab').forEach(b=>b.addEventListener('click',()=>{let k=b.dataset.layer;state.layers[k]=!state.layers[k];b.classList.toggle('active',state.layers[k]);b.setAttribute('aria-pressed',String(state.layers[k]));b.querySelector('.toggle').classList.toggle('on',state.layers[k]);render();}));
  document.getElementById('timeline').addEventListener('input',e=>{stopReplay();state.vesselProgress=Number(e.target.value);const day=Math.floor(state.vesselProgress);if(day!==state.day){state.day=day;render();}else updateVesselPosition();});
  document.getElementById('compare-toggle').addEventListener('click',e=>{state.compare=!state.compare;e.currentTarget.classList.toggle('on',state.compare);e.currentTarget.setAttribute('aria-checked',String(state.compare));render();});
  document.getElementById('legend-toggle').addEventListener('click',e=>{const l=document.querySelector('.map-legend');l.classList.toggle('collapsed');const expanded=!l.classList.contains('collapsed');e.currentTarget.textContent=expanded?'−':'+';e.currentTarget.setAttribute('aria-expanded',String(expanded));e.currentTarget.setAttribute('aria-label',expanded?'Collapse map key':'Expand map key');});
  document.getElementById('zoom-in').addEventListener('click',()=>{state.zoom=Math.min(1.4,state.zoom+.1);svg.style.transform=`scale(${state.zoom})`;svg.style.transformOrigin='50% 50%';document.querySelector('.zoom-controls span').textContent=`${state.zoom.toFixed(1)}×`;});
  document.getElementById('zoom-out').addEventListener('click',()=>{state.zoom=Math.max(.8,state.zoom-.1);svg.style.transform=`scale(${state.zoom})`;svg.style.transformOrigin='50% 50%';document.querySelector('.zoom-controls span').textContent=`${state.zoom.toFixed(1)}×`;});
  document.getElementById('fullscreen-button').addEventListener('click',()=>{document.getElementById('map-canvas').classList.toggle('expanded');});
  document.getElementById('about-button').addEventListener('click',()=>document.getElementById('info-dialog').showModal());
  document.getElementById('play-button').addEventListener('click',()=>{if(state.playing)stopReplay();else startReplay();});
  document.getElementById('map-canvas').addEventListener('click',e=>{if(e.target===e.currentTarget&&e.currentTarget.classList.contains('expanded'))e.currentTarget.classList.remove('expanded');});
  document.getElementById('observations-button').addEventListener('click',()=>{if(observedView){showScenario();return;}const items=observationCatalog?.observations||[];if(items.length)showObservation(document.getElementById('observation-date').value||items.at(-1).date);});
  const scrollToWorkspace=()=>{
    const workspace=document.getElementById('workspace');
    const top=window.scrollY+workspace.getBoundingClientRect().top;
    window.scrollTo({top,behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
  };
  document.getElementById('nav-scenario').addEventListener('click',()=>{if(observedView||document.querySelector('.app-shell').dataset.workArea==='reviews')showScenario();else setWorkArea('scenario');scrollToWorkspace();});
  document.getElementById('nav-observations').addEventListener('click',()=>{const items=observationCatalog?.observations||[];if(!items.length)return;const selected=document.getElementById('observation-date').value||items.at(-1).date;if(observedView&&currentObservedDate===selected){setWorkArea('observations');}else showObservation(selected);scrollToWorkspace();});
  document.getElementById('nav-reviews').addEventListener('click',()=>{setWorkArea('reviews');document.getElementById('review-workflow-panel').open=true;scrollToWorkspace();});
  document.getElementById('replay-observation-button').addEventListener('click',()=>{document.getElementById('observations-button').click();document.querySelector('.workspace-head').scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});});
  document.getElementById('observation-date').addEventListener('change',e=>{if(observedView)showObservation(e.currentTarget.value);});
  document.getElementById('observation-forecast-toggle').addEventListener('click',showResearchForecast);
  document.getElementById('route-candidate-form').addEventListener('submit',submitRouteCandidates);
  document.getElementById('route-exposure-form').addEventListener('submit',submitRouteExposure);
  document.getElementById('route-exposure-form').addEventListener('input',()=>{if(routeReviewInput)clearRouteExposure(true);});
  document.getElementById('route-analysis-half-width').addEventListener('change',()=>{if(routeReviewInput)clearRouteExposure(true);});
  document.getElementById('route-export-button').addEventListener('click',downloadRouteReviewPacket);
  document.getElementById('route-geojson-button').addEventListener('click',downloadRouteGeoJSON);
  render();
  loadObservationCatalog();
  loadObservedIcebergTracks();
  showResearchModelStatus();
  loadTransectEvidence();
  loadMotionEvidence();
  const evidence=document.querySelector('.replay-evidence');
  if('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches){
    const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting){entry.target.classList.add('in-view');observer.unobserve(entry.target);}}},{threshold:.18});
    observer.observe(evidence);
  }
  const finePointer=window.matchMedia('(pointer: fine) and (hover: hover)');
  const reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)');
  const cursor=document.querySelector('.aesthetic-cursor');
  if(cursor&&finePointer.matches&&!reducedMotion.matches){
    document.documentElement.classList.add('custom-cursor-enabled');
    let x=-100,y=-100,px=-100,py=-100,raf=0;
    const animateCursor=()=>{px+=(x-px)*.24;py+=(y-py)*.24;cursor.style.transform=`translate3d(${px}px, ${py}px, 0)`;if(Math.abs(x-px)+Math.abs(y-py)>.25)raf=requestAnimationFrame(animateCursor);else raf=0;};
    document.addEventListener('pointermove',event=>{if(event.pointerType!=='mouse')return;x=event.clientX;y=event.clientY;cursor.classList.add('visible');if(!raf)raf=requestAnimationFrame(animateCursor);},{passive:true});
    document.addEventListener('pointerover',event=>cursor.classList.toggle('interactive',Boolean(event.target.closest('a, button, input, select, textarea, summary, [role="switch"]'))));
    document.addEventListener('pointerdown',()=>cursor.classList.add('pressed'));
    document.addEventListener('pointerup',()=>cursor.classList.remove('pressed'));
    document.addEventListener('pointerleave',()=>cursor.classList.remove('visible'));
    window.addEventListener('blur',()=>cursor.classList.remove('visible'));
  }
})();
