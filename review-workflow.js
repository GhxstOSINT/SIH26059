(() => {
  'use strict';
  const panel=document.getElementById('review-workflow-panel');
  const stateNode=document.getElementById('review-workflow-state');
  const message=document.getElementById('review-workflow-message');
  const body=document.getElementById('review-workflow-body');
  const actorNode=document.getElementById('review-actor');
  const listNode=document.getElementById('review-case-list');
  const detailNode=document.getElementById('review-case-detail');
  const form=document.getElementById('review-decision-form');
  const fileButton=document.getElementById('file-case-button');
  const formStatus=document.getElementById('review-submit-status');
  const gateSelect=document.getElementById('review-gate');
  const decisionSelect=document.getElementById('review-decision');
  const gateLabels={science:'Science',maritime:'Maritime safety',security:'Security',operations:'Operations',handoff:'Release handoff'};
  const gateRoles={science:'science_reviewer',maritime:'maritime_reviewer',security:'security_reviewer',operations:'operations_reviewer',handoff:'release_authority'};
  let actor=null, route=null, sourceHash=null, selectedCase=null, busy=false;
  const apiRoot=()=>{
    const local=['localhost','127.0.0.1','::1'].includes(location.hostname);
    const defaultRoot=local&&!location.pathname.startsWith('/portal')?'http://127.0.0.1:8000':location.origin;
    return String(window.SOUTHERN_PASSAGE_API_URL||(local?new URLSearchParams(location.search).get('api'):null)||defaultRoot).replace(/\/$/,'');
  };
  const endpoint='/api/v1/review-workflow';
  async function api(path, options={}){
    const controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),30000);
    try{
      const response=await fetch(`${apiRoot()}${endpoint}${path}`,{credentials:'include',signal:controller.signal,
        headers:{Accept:'application/json',...(options.body?{'Content-Type':'application/json'}:{})},...options});
      const data=await response.json().catch(()=>({}));
      if(!response.ok){const error=new Error(typeof data.detail==='string'?data.detail:`Request failed (${response.status}).`);error.status=response.status;throw error;}
      return data;
    }finally{clearTimeout(timeout);}
  }
  function errorText(error){
    if(error.status===401)return 'The gateway session is missing or expired. Sign in through the ministry portal and retry.';
    if(error.status===403)return 'Your assigned role cannot perform this action. Ask the portal administrator to check the role mapping.';
    if(error.status===409)return error.message||'This case changed. Refresh and review it again.';
    if(error.status===422)return error.message||'Check the form values and try again.';
    if(error.status===503)return 'Review storage or identity configuration is unavailable. Contact the service administrator.';
    if(error.name==='AbortError')return 'The review service timed out. Retry after checking the connection.';
    return 'The review service could not be reached. Check the connection and retry.';
  }
  const node=(tag,className,text)=>{const item=document.createElement(tag);if(className)item.className=className;if(text!==undefined)item.textContent=text;return item;};
  function updateFileButton(){fileButton.disabled=busy||!route||!sourceHash||!actor?.roles?.includes('analyst');
    fileButton.title=!actor?'Ministry SSO review access is not connected.':!actor.roles.includes('analyst')?'The analyst role is required.':!route?'Review a dated route first.':'';}
  function displayStatus(){stateNode.textContent=actor?'Connected':body.hidden?'Not connected':'Available';updateFileButton();}
  function renderList(items){
    listNode.replaceChildren();
    listNode.append(node('h3','',`Cases · ${items.length}`));
    if(!items.length){listNode.append(node('p','review-empty','No cases yet. An analyst can review a dated route and file its evidence packet.'));return;}
    for(const item of items){
      const button=node('button','review-case-item');button.type='button';button.dataset.caseId=item.case_id;
      button.append(node('strong','',item.route_label||item.analysis_key.slice(0,12)),
                    node('span','',`${item.status.replaceAll('_',' ')} · ${new Date(item.created_at).toLocaleDateString()}`));
      button.setAttribute('aria-pressed',String(item.case_id===selectedCase));
      button.addEventListener('click',()=>openCase(item.case_id));listNode.append(button);
    }
  }
  async function refreshCases(){
    if(!actor)return;
    const button=document.getElementById('review-refresh');button.disabled=true;
    try{const result=await api('/cases');renderList(result.items||[]);if(selectedCase)await openCase(selectedCase);}
    catch(error){message.textContent=errorText(error);}
    finally{button.disabled=false;}
  }
  function eligibleGates(record){
    if(!actor||record.creator?.sub===actor.sub&&record.creator?.iss===actor.iss||record.events?.some(event=>event.actor_subject===actor.sub&&event.actor_issuer===actor.iss))return [];
    if(['changes_required','pilot_review_packaged'].includes(record.status))return [];
    return Object.keys(gateRoles).filter(gate=>actor.roles.includes(gateRoles[gate])&&!record.decisions?.[gate]
      &&(gate!=='handoff'||record.status==='awaiting_release_handoff'));
  }
  function renderDetail(record){
    detailNode.replaceChildren();
    detailNode.append(node('h3','',record.packet?.route?.route_id||`Case ${record.case_id.slice(0,8)}`));
    detailNode.append(node('p','review-case-meta',`Filed ${new Date(record.created_at).toLocaleString()} · ${record.status.replaceAll('_',' ')}`));
    detailNode.append(node('p','review-case-meta',`Evidence SHA-256 ${record.packet_sha256}`));
    const download=node('button','review-download','Download case record');download.type='button';
    download.addEventListener('click',()=>{
      const url=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)+'\n'],{type:'application/json'}));
      const link=document.createElement('a');link.href=url;link.download=`southern-passage-case-${record.case_id}.json`;
      document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
    });detailNode.append(download);
    const gates=node('div','review-gates');
    for(const gate of ['science','maritime','security','operations','handoff']){
      const event=record.events?.find(entry=>entry.gate===gate);
      const row=node('div','review-gate');
      row.append(node('strong','',gateLabels[gate]),node('span',event?'Recorded: '+event.decision.replaceAll('_',' '):'Awaiting review'));
      if(event){row.append(node('small','',`${event.actor_subject} · ${new Date(event.created_at).toLocaleString()}`));
        row.append(node('p','',event.rationale));
        for(const evidence of event.evidence||[])row.append(node('small','',`Evidence ${evidence.record_id} · SHA-256 ${evidence.sha256}`));}
      gates.append(row);
    }
    detailNode.append(gates,node('p','review-boundary',record.warning));
    const allowed=eligibleGates(record);
    gateSelect.replaceChildren();
    for(const gate of allowed){const option=node('option','',gateLabels[gate]);option.value=gate;gateSelect.append(option);}
    form.hidden=!allowed.length;
    if(allowed.length)syncDecisionOptions();
    document.querySelectorAll('.review-case-item').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.caseId===record.case_id)));
  }
  function syncDecisionOptions(){
    const handoff=gateSelect.value==='handoff';
    decisionSelect.replaceChildren();
    for(const value of handoff?['package']:['reviewed','blocked','changes_required']){
      const option=node('option','',value.replaceAll('_',' '));option.value=value;decisionSelect.append(option);
    }
  }
  async function openCase(caseId){
    selectedCase=caseId;detailNode.replaceChildren(node('p','','Loading the case record…'));form.hidden=true;
    try{renderDetail(await api(`/cases/${encodeURIComponent(caseId)}`));}
    catch(error){detailNode.replaceChildren(node('p','review-error',errorText(error)));}
  }
  async function loadAccess(){
    try{
      const capabilities=await api('/capabilities');
      if(!capabilities.enabled){message.textContent='The ministry SSO gateway and review ledger are not connected in this environment. Observation analysis remains available.';stateNode.textContent='Not connected';return;}
      actor=await api('/me');body.hidden=false;actorNode.textContent=`Signed in as ${actor.sub} · ${actor.roles.join(', ')}`;
      message.textContent='Independent reviewers can record evidence-bound process attestations. No gate grants navigation clearance.';
      displayStatus();await refreshCases();
    }catch(error){actor=null;body.hidden=true;stateNode.textContent='Sign-in required';message.textContent=errorText(error);updateFileButton();}
  }
  async function fileCase(){
    if(!route||!sourceHash||!actor?.roles.includes('analyst')||busy)return;
    busy=true;updateFileButton();fileButton.textContent='Filing…';
    const currentRoute=route,currentHash=sourceHash;
    try{
      const result=await api('/cases',{method:'POST',body:JSON.stringify({route:currentRoute,expected_source_sha256:currentHash})});
      panel.open=true;selectedCase=result.case_id;message.textContent='Case filed with a frozen evidence packet. Independent gate reviews are now required.';
      await refreshCases();
    }catch(error){panel.open=true;message.textContent=errorText(error);}
    finally{busy=false;fileButton.textContent='File for review';updateFileButton();}
  }
  async function recordDecision(event){
    event.preventDefault();if(!selectedCase||busy)return;
    const gate=gateSelect.value,decision=decisionSelect.value;
    const recordId=document.getElementById('review-record-id').value.trim();
    const sha=document.getElementById('review-record-sha').value.trim();
    if((recordId&&!sha)||(!recordId&&sha)||(['reviewed','package'].includes(decision)&&!recordId)){
      formStatus.textContent='Provide both an approved record ID and its SHA-256 for an affirmative attestation.';return;
    }
    const rationale=document.getElementById('review-rationale').value.trim();
    if(rationale.length<30){formStatus.textContent='Explain the review in at least 30 characters.';return;}
    const evidence=recordId?[{record_id:recordId,sha256:sha}]:[];
    busy=true;document.getElementById('review-submit').disabled=true;formStatus.textContent='Recording this immutable decision…';
    try{
      const record=await api(`/cases/${encodeURIComponent(selectedCase)}/decisions`,{method:'POST',body:JSON.stringify({gate,decision,rationale,evidence})});
      form.reset();formStatus.textContent='Decision recorded. It cannot be edited; file a new case if evidence changes.';
      renderDetail(record);await refreshCases();
    }catch(error){formStatus.textContent=errorText(error);}
    finally{busy=false;document.getElementById('review-submit').disabled=false;}
  }
  window.addEventListener('southernpassage:route-reviewed',event=>{route=event.detail.request;sourceHash=event.detail.sourceHash;updateFileButton();});
  window.addEventListener('southernpassage:route-cleared',()=>{route=null;sourceHash=null;updateFileButton();});
  fileButton.addEventListener('click',fileCase);
  document.getElementById('review-refresh').addEventListener('click',refreshCases);
  gateSelect.addEventListener('change',syncDecisionOptions);
  form.addEventListener('submit',recordDecision);
  loadAccess();
})();
