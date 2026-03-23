import { useState, useCallback } from "react";

// Chess Engine — Built with DeepResearch Level 3 Protocol
const E=0,WP=1,WN=2,WB=3,WR=4,WQ=5,WK=6,BP=7,BN=8,BB=9,BR=10,BQ=11,BK=12;
const PC={[WP]:"♙",[WN]:"♘",[WB]:"♗",[WR]:"♖",[WQ]:"♕",[WK]:"♔",[BP]:"♟",[BN]:"♞",[BB]:"♝",[BR]:"♜",[BQ]:"♛",[BK]:"♚"};
const isW=p=>p>=1&&p<=6,isB=p=>p>=7&&p<=12,col=p=>p===0?0:isW(p)?1:2;
const PV={[WP]:100,[WN]:320,[WB]:330,[WR]:500,[WQ]:900,[WK]:20000,[BP]:-100,[BN]:-320,[BB]:-330,[BR]:-500,[BQ]:-900,[BK]:-20000};
const PST={[WP]:[0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0],[WN]:[-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50],[WB]:[-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20],[WR]:[0,0,0,0,0,0,0,0,5,10,10,10,10,10,10,5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0],[WQ]:[-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20],[WK]:[-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10,20,20,0,0,0,0,20,20,20,30,10,0,0,10,30,20]};
const mi=i=>(7-Math.floor(i/8))*8+(i%8);
const pst=(p,s)=>{const b=p<=6?p:p-6;const t=PST[b];return t?(p<=6?t[s]:-t[mi(s)]):0};
function initB(){const b=Array(64).fill(0);[BR,BN,BB,BQ,BK,BB,BN,BR].forEach((p,i)=>b[i]=p);for(let i=8;i<16;i++)b[i]=BP;for(let i=48;i<56;i++)b[i]=WP;[WR,WN,WB,WQ,WK,WB,WN,WR].forEach((p,i)=>b[56+i]=p);return b}
function initS(){return{b:initB(),t:1,c:{wk:1,wq:1,bk:1,bq:1},ep:-1,h:0}}

function isAtt(b,sq,by){
  const r=sq>>3,c=sq&7;
  for(const[dr,dc]of[[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]){const nr=r+dr,nc=c+dc;if(nr>=0&&nr<8&&nc>=0&&nc<8){const p=b[nr*8+nc];if(p&&col(p)===by&&(p<=6?p:p-6)===2)return 1}}
  for(const[dr,dc]of[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]){const nr=r+dr,nc=c+dc;if(nr>=0&&nr<8&&nc>=0&&nc<8){const p=b[nr*8+nc];if(p&&col(p)===by&&(p<=6?p:p-6)===6)return 1}}
  const pd=by===1?-1:1;for(const dc of[-1,1]){const nr=r+pd,nc=c+dc;if(nr>=0&&nr<8&&nc>=0&&nc<8){const p=b[nr*8+nc];if(p&&col(p)===by&&(p<=6?p:p-6)===1)return 1}}
  for(const[dr,dc]of[[-1,-1],[-1,1],[1,-1],[1,1]]){let nr=r+dr,nc=c+dc;while(nr>=0&&nr<8&&nc>=0&&nc<8){const p=b[nr*8+nc];if(p){if(col(p)===by){const x=p<=6?p:p-6;if(x===3||x===5)return 1}break}nr+=dr;nc+=dc}}
  for(const[dr,dc]of[[-1,0],[1,0],[0,-1],[0,1]]){let nr=r+dr,nc=c+dc;while(nr>=0&&nr<8&&nc>=0&&nc<8){const p=b[nr*8+nc];if(p){if(col(p)===by){const x=p<=6?p:p-6;if(x===4||x===5)return 1}break}nr+=dr;nc+=dc}}
  return 0
}
function fK(b,cl){const k=cl===1?WK:BK;for(let i=0;i<64;i++)if(b[i]===k)return i;return-1}

function pMoves(s,fc){
  const{b,c,ep}=s,mv=[];
  for(let sq=0;sq<64;sq++){
    const p=b[sq];if(!p||col(p)!==fc)continue;
    const r=sq>>3,cc=sq&7,x=p<=6?p:p-6;
    if(x===1){
      const d=fc===1?-1:1,sr=fc===1?6:1,pr=fc===1?0:7,f=sq+d*8;
      if(f>=0&&f<64&&!b[f]){if((f>>3)===pr){for(const pp of fc===1?[WQ,WR,WB,WN]:[BQ,BR,BB,BN])mv.push({f:sq,t:f,p:pp})}else{mv.push({f:sq,t:f});if(r===sr){const f2=sq+d*16;if(!b[f2])mv.push({f:sq,t:f2,d:1})}}}
      for(const dc of[-1,1]){const nc2=cc+dc;if(nc2<0||nc2>7)continue;const cs=sq+d*8+dc;if(cs<0||cs>63)continue;if(b[cs]&&col(b[cs])!==fc){if((cs>>3)===pr){for(const pp of fc===1?[WQ,WR,WB,WN]:[BQ,BR,BB,BN])mv.push({f:sq,t:cs,p:pp})}else mv.push({f:sq,t:cs})}if(cs===ep)mv.push({f:sq,t:cs,e:1})}
    }else if(x===2){for(const[dr,dc]of[[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]){const nr=r+dr,nc=cc+dc;if(nr>=0&&nr<8&&nc>=0&&nc<8){const to=nr*8+nc;if(!b[to]||col(b[to])!==fc)mv.push({f:sq,t:to})}}}
    else if(x===3||x===4||x===5){const dirs=x===3?[[-1,-1],[-1,1],[1,-1],[1,1]]:x===4?[[-1,0],[1,0],[0,-1],[0,1]]:[[-1,-1],[-1,1],[1,-1],[1,1],[-1,0],[1,0],[0,-1],[0,1]];for(const[dr,dc]of dirs){let nr=r+dr,nc=cc+dc;while(nr>=0&&nr<8&&nc>=0&&nc<8){const to=nr*8+nc;if(b[to]){if(col(b[to])!==fc)mv.push({f:sq,t:to});break}mv.push({f:sq,t:to});nr+=dr;nc+=dc}}}
    else if(x===6){
      for(const[dr,dc]of[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]){const nr=r+dr,nc=cc+dc;if(nr>=0&&nr<8&&nc>=0&&nc<8){const to=nr*8+nc;if(!b[to]||col(b[to])!==fc)mv.push({f:sq,t:to})}}
      if(fc===1){if(c.wk&&!b[61]&&!b[62]&&b[60]===WK&&b[63]===WR&&!isAtt(b,60,2)&&!isAtt(b,61,2)&&!isAtt(b,62,2))mv.push({f:60,t:62,cs:'wk'});if(c.wq&&!b[59]&&!b[58]&&!b[57]&&b[60]===WK&&b[56]===WR&&!isAtt(b,60,2)&&!isAtt(b,59,2)&&!isAtt(b,58,2))mv.push({f:60,t:58,cs:'wq'})}
      else{if(c.bk&&!b[5]&&!b[6]&&b[4]===BK&&b[7]===BR&&!isAtt(b,4,1)&&!isAtt(b,5,1)&&!isAtt(b,6,1))mv.push({f:4,t:6,cs:'bk'});if(c.bq&&!b[3]&&!b[2]&&!b[1]&&b[4]===BK&&b[0]===BR&&!isAtt(b,4,1)&&!isAtt(b,3,1)&&!isAtt(b,2,1))mv.push({f:4,t:2,cs:'bq'})}
    }
  }
  return mv
}

function mk(s,m){
  const b=[...s.b],c={...s.c};let ep=-1;
  if(m.e)b[m.t+(s.t===1?8:-8)]=0;
  if(m.cs){if(m.cs==='wk'){b[61]=WR;b[63]=0}if(m.cs==='wq'){b[59]=WR;b[56]=0}if(m.cs==='bk'){b[5]=BR;b[7]=0}if(m.cs==='bq'){b[3]=BR;b[0]=0}}
  b[m.t]=m.p||b[m.f];b[m.f]=0;if(m.d)ep=(m.f+m.t)/2;
  if(m.f===60||m.t===60){c.wk=0;c.wq=0}if(m.f===4||m.t===4){c.bk=0;c.bq=0}
  if(m.f===63||m.t===63)c.wk=0;if(m.f===56||m.t===56)c.wq=0;if(m.f===7||m.t===7)c.bk=0;if(m.f===0||m.t===0)c.bq=0;
  return{b,t:s.t===1?2:1,c,ep,h:s.h+1}
}

function legal(s){return pMoves(s,s.t).filter(m=>{const n=mk(s,m);const k=fK(n.b,s.t);return k>=0&&!isAtt(n.b,k,s.t===1?2:1)})}
function inCk(s){const k=fK(s.b,s.t);return k>=0&&isAtt(s.b,k,s.t===1?2:1)}

function ev(b){let s=0;for(let i=0;i<64;i++){const p=b[i];if(p)s+=(PV[p]||0)+pst(p,i)}return s}
function ord(b,mvs){return mvs.map(m=>{let s=0;if(b[m.t])s+=10*Math.abs(PV[b[m.t]]||0)-Math.abs(PV[b[m.f]]||0);if(m.p)s+=Math.abs(PV[m.p]||0);return{...m,_s:s}}).sort((a,b)=>b._s-a._s)}

function ab(s,d,a,bt,mx){
  if(d===0)return ev(s.b);
  const mvs=legal(s);if(!mvs.length)return inCk(s)?(mx?-99999+s.h:99999-s.h):0;
  const o=ord(s.b,mvs);
  if(mx){let best=-1e9;for(const m of o){const v=ab(mk(s,m),d-1,a,bt,0);best=Math.max(best,v);a=Math.max(a,v);if(bt<=a)break}return best}
  else{let best=1e9;for(const m of o){const v=ab(mk(s,m),d-1,a,bt,1);best=Math.min(best,v);bt=Math.min(bt,v);if(bt<=a)break}return best}
}

function bestMove(s,depth=3){
  const mvs=legal(s);if(!mvs.length)return null;
  const mx=s.t===1;let bm=mvs[0],bv=mx?-1e9:1e9;
  const o=ord(s.b,mvs);
  for(const m of o){const v=ab(mk(s,m),depth-1,-1e9,1e9,!mx);if(mx?v>bv:v<bv){bv=v;bm=m}}
  const th=mx?bv-15:bv+15;
  const good=o.filter(m=>{const v=ab(mk(s,m),depth-1,-1e9,1e9,!mx);return mx?v>=th:v<=th});
  return good.length>1?good[Math.floor(Math.random()*good.length)]:bm
}

const SN=i=>String.fromCharCode(97+(i&7))+(8-(i>>3));

export default function Chess(){
  const[s,setS]=useState(initS);
  const[sel,setSel]=useState(-1);
  const[lg,setLg]=useState([]);
  const[th,setTh]=useState(0);
  const[go,setGo]=useState(null);
  const[dp,setDp]=useState(3);
  const[ml,setMl]=useState([]);
  const[lm,setLm]=useState(null);
  const[ps,setPs]=useState(null);
  const[pf,setPf]=useState(null);
  const[cap,setCap]=useState({w:[],b:[]});

  const chkEnd=useCallback(st=>{const m=legal(st);if(!m.length){setGo(inCk(st)?(st.t===1?"Schwarz gewinnt! ♚":"Weiß gewinnt! ♔"):"Patt! ½-½");return 1}return 0},[]);

  const doAi=useCallback(st=>{
    setTh(1);
    setTimeout(()=>{
      const m=bestMove(st,dp);
      if(m){
        const captured=st.b[m.t];
        const n=mk(st,m);setS(n);setLm(m);
        if(captured)setCap(prev=>({...prev,w:[...prev.w,captured]}));
        setMl(p=>[...p,`${PC[st.b[m.f]]}${SN(m.t)}`]);chkEnd(n)
      }
      setTh(0)
    },50)
  },[dp,chkEnd]);

  const click=useCallback(sq=>{
    if(th||go||s.t!==1)return;
    if(sel>=0){
      const m=lg.find(x=>x.t===sq);
      if(m){
        const base=s.b[sel]<=6?s.b[sel]:s.b[sel]-6;
        if(base===1&&((sq>>3)===0||(sq>>3)===7)){setPs(sq);setPf(sel);setSel(-1);setLg([]);return}
        const captured=s.b[m.t];
        const n=mk(s,m);setS(n);setLm(m);
        if(captured)setCap(prev=>({...prev,b:[...prev.b,captured]}));
        setMl(p=>[...p,`${PC[s.b[m.f]]}${SN(m.t)}`]);setSel(-1);setLg([]);
        if(!chkEnd(n))doAi(n);return
      }
    }
    if(s.b[sq]&&col(s.b[sq])===1){setSel(sq);setLg(legal(s).filter(x=>x.f===sq))}else{setSel(-1);setLg([])}
  },[s,sel,lg,th,go,doAi,chkEnd]);

  const promo=useCallback(pc=>{
    const m=legal(s).find(x=>x.f===pf&&x.t===ps&&x.p===pc);
    if(m){const n=mk(s,m);setS(n);setLm(m);setMl(p=>[...p,`♙${SN(ps)}=${PC[pc]}`]);if(!chkEnd(n))doAi(n)}
    setPs(null);setPf(null)
  },[s,ps,pf,doAi,chkEnd]);

  const reset=()=>{setS(initS());setSel(-1);setLg([]);setTh(0);setGo(null);setMl([]);setLm(null);setPs(null);setPf(null);setCap({w:[],b:[]})};

  const ck=inCk(s)?fK(s.b,s.t):-1;
  const ls=new Set(lg.map(x=>x.t));

  return(
    <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:14,padding:"12px 0",fontFamily:"'Georgia',serif"}}>
      <div style={{display:"flex",alignItems:"baseline",gap:10}}>
        <span style={{fontSize:22,fontWeight:700,color:"var(--color-text-primary)",letterSpacing:"-0.5px"}}>DeepResearch Chess</span>
        <span style={{fontSize:11,color:"var(--color-text-tertiary)"}}>Tiefe {dp}</span>
      </div>

      {/* Captured pieces */}
      <div style={{display:"flex",justifyContent:"space-between",width:368,fontSize:16,minHeight:22}}>
        <div style={{color:"var(--color-text-secondary)"}}>{cap.b.sort((a,b)=>Math.abs(PV[a])-Math.abs(PV[b])).map((p,i)=><span key={i}>{PC[p]}</span>)}</div>
        <div style={{color:"var(--color-text-secondary)"}}>{cap.w.sort((a,b)=>Math.abs(PV[a])-Math.abs(PV[b])).map((p,i)=><span key={i}>{PC[p]}</span>)}</div>
      </div>

      <div style={{position:"relative",width:368,height:368,borderRadius:6,overflow:"hidden",boxShadow:"0 8px 32px rgba(0,0,0,0.25)",border:"2px solid var(--color-border-secondary)"}}>
        {s.b.map((p,i)=>{
          const r=i>>3,c=i&7,lt=(r+c)%2===0;
          let bg=lt?"#f0d9b5":"#b58863";
          if(i===sel)bg="#829769";
          if(lm&&(i===lm.f||i===lm.t))bg=lt?"#e8e086":"#c8a836";
          if(i===ck)bg="#e04040";
          return(
            <div key={i} onClick={()=>click(i)} style={{
              position:"absolute",left:c*46,top:r*46,width:46,height:46,background:bg,
              display:"flex",alignItems:"center",justifyContent:"center",
              cursor:(p&&col(p)===1&&s.t===1&&!th&&!go)||ls.has(i)?"pointer":"default",
              userSelect:"none",fontSize:34,lineHeight:1,transition:"background 0.15s"
            }}>
              {ls.has(i)&&!p&&<div style={{width:13,height:13,borderRadius:"50%",background:"rgba(0,0,0,0.18)"}}/>}
              {ls.has(i)&&p&&<div style={{position:"absolute",inset:0,border:"3.5px solid rgba(0,0,0,0.3)",borderRadius:2}}/>}
              {p?<span style={{color:isB(p)?"#222":"#fff",textShadow:isB(p)?"0 0 2px rgba(255,255,255,0.3)":"0 1px 3px rgba(0,0,0,0.5)",pointerEvents:"none"}}>{PC[p]}</span>:null}
              {c===0&&<span style={{position:"absolute",top:2,left:3,fontSize:9,fontWeight:700,color:lt?"#b58863":"#f0d9b5",fontFamily:"system-ui"}}>{8-r}</span>}
              {r===7&&<span style={{position:"absolute",bottom:1,right:3,fontSize:9,fontWeight:700,color:lt?"#b58863":"#f0d9b5",fontFamily:"system-ui"}}>{String.fromCharCode(97+c)}</span>}
            </div>
          )
        })}
        {ps!==null&&<div style={{position:"absolute",inset:0,background:"rgba(0,0,0,0.55)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:10}}>
          <div style={{background:"var(--color-background-primary)",borderRadius:10,padding:16,display:"flex",gap:8,boxShadow:"0 8px 32px rgba(0,0,0,0.4)"}}>
            {[WQ,WR,WB,WN].map(pp=><button key={pp} onClick={()=>promo(pp)} style={{fontSize:38,width:54,height:54,border:"1px solid var(--color-border-tertiary)",borderRadius:8,background:"var(--color-background-secondary)",cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center"}}>{PC[pp]}</button>)}
          </div>
        </div>}
        {th&&<div style={{position:"absolute",bottom:8,left:8,background:"rgba(0,0,0,0.7)",color:"#fff",padding:"4px 10px",borderRadius:4,fontSize:12,fontFamily:"system-ui"}}>Denkt...</div>}
        {go&&<div style={{position:"absolute",inset:0,background:"rgba(0,0,0,0.5)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:5}}>
          <div style={{background:"var(--color-background-primary)",borderRadius:12,padding:"20px 28px",textAlign:"center",boxShadow:"0 8px 32px rgba(0,0,0,0.4)"}}>
            <div style={{fontSize:20,fontWeight:700,color:"var(--color-text-primary)",marginBottom:8}}>{go}</div>
            <button onClick={reset} style={{padding:"8px 20px",borderRadius:8,border:"none",background:"#b58863",color:"#fff",cursor:"pointer",fontSize:14,fontWeight:600}}>Neues Spiel</button>
          </div>
        </div>}
      </div>

      <div style={{display:"flex",gap:10,alignItems:"center"}}>
        <button onClick={reset} style={{padding:"6px 16px",borderRadius:6,border:"1px solid var(--color-border-tertiary)",background:"var(--color-background-secondary)",color:"var(--color-text-primary)",cursor:"pointer",fontSize:13,fontWeight:500}}>Neues Spiel</button>
        <select value={dp} onChange={e=>setDp(+e.target.value)} style={{padding:"5px 8px",borderRadius:5,border:"1px solid var(--color-border-tertiary)",background:"var(--color-background-secondary)",color:"var(--color-text-primary)",fontSize:12}}>
          <option value={2}>Leicht</option><option value={3}>Mittel</option><option value={4}>Stark</option>
        </select>
      </div>

      {ml.length>0&&<div style={{maxHeight:80,overflowY:"auto",fontSize:11,color:"var(--color-text-secondary)",width:368,padding:"6px 10px",background:"var(--color-background-secondary)",borderRadius:6,lineHeight:2,fontFamily:"system-ui"}}>
        {ml.map((m,i)=><span key={i} style={{marginRight:6}}>{i%2===0?`${Math.floor(i/2)+1}. `:""}{m}{i%2===0?" ":""}</span>)}
      </div>}
    </div>
  )
}
