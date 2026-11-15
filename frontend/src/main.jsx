import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Bookmark, Bot, ChevronDown, CircleHelp, GitCompareArrows, Heart,
  History, LayoutGrid, Menu, Plus, Search, Send, Settings, ShieldCheck,
  SlidersHorizontal, Sparkles, Star, Tag, X
} from 'lucide-react'
import './styles.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const SAVED_KEY = 'shopsmart-saved-products'
const CHATS_KEY = 'shopsmart-chats'
const ACTIVE_CHAT_KEY = 'shopsmart-active-chat'
const LEGACY_HISTORY_KEY = 'shopsmart-history'
const suggestions = [
  'Find a laptop under ₹60,000 for programming',
  'Suggest wireless headphones under ₹5,000',
  'Find a practical birthday gift under ₹3,000',
  'Compare office chairs for long working hours',
]
const categories = ['All categories', 'Electronics', 'Fashion and Clothing', 'Home and Kitchen', 'Beauty and Personal Care', 'Books and Stationery', 'Sports and Fitness', 'Furniture and Home Decor', 'Travel and Luggage', 'Pet Supplies']

function newChatRecord() {
  return { id: `web-${crypto.randomUUID()}`, title: 'New chat', messages: [], updatedAt: new Date().toISOString() }
}

function readChatState() {
  try {
    const stored = JSON.parse(localStorage.getItem(CHATS_KEY) || '[]')
    let chats = Array.isArray(stored) ? stored : []
    if (chats.length === 0) {
      const legacyMessages = JSON.parse(localStorage.getItem(LEGACY_HISTORY_KEY) || '[]')
      if (Array.isArray(legacyMessages) && legacyMessages.length > 0) {
        const migrated = newChatRecord()
        migrated.messages = legacyMessages
        migrated.title = legacyMessages.find((message) => message.role === 'user')?.text?.slice(0, 42) || 'Previous chat'
        chats = [migrated]
      } else {
        chats = [newChatRecord()]
      }
    }
    const activeId = localStorage.getItem(ACTIVE_CHAT_KEY)
    const active = chats.find((chat) => chat.id === activeId) || chats[0]
    return { chats, active }
  } catch {
    const active = newChatRecord()
    return { chats: [active], active }
  }
}

function money(value) {
  if (value === null || value === undefined) return 'Price unavailable'
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(value)
}

function ProductCard({ product, saved, selected, onSave, onCompare }) {
  const image = product.image_url
  return <article className="product-card">
    <div className="product-image">
      {image ? <img src={image} alt={product.name} /> : <div className="image-placeholder"><LayoutGrid size={24} /></div>}
      <button className={`save-product ${saved ? 'is-saved' : ''}`} onClick={() => onSave(product)} aria-label={saved ? 'Remove saved product' : 'Save product'}><Heart size={16} fill={saved ? 'currentColor' : 'none'} /></button>
      {product.discount_percent > 0 && <span className="discount"><Tag size={12} /> {product.discount_percent}% off</span>}
    </div>
    <div className="product-copy">
      <div className="product-meta"><span>{product.source_name || 'External source'}</span><span>{product.subcategory || product.category}</span></div>
      <h3>{product.name}</h3>
      {product.brand && <p className="brand-line">{product.brand}</p>}
          <div className="product-rating">{product.rating > 0 && product.review_count >= 5 ? <><Star size={14} fill="currentColor" /> {product.rating} <span>({product.review_count || 0})</span></> : <span>Rating unavailable</span>}</div>
          <div className="product-price"><strong>{money(product.price)}</strong>{product.original_price && <del>{money(product.original_price)}</del>}</div><div className="verification-label">{product.price_verified ? 'Price verified' : 'Price not verified'} · {product.availability === 'out_of_stock' ? 'Out of stock' : product.availability === 'in_stock' && product.availability_verified ? 'Stock confirmed' : 'Availability not confirmed'}</div>
      {product.rank_explanation ? <details className="rank-details">
        <summary>Why this rank? <span>Recommended #{product.rank}</span></summary>
        <div className="rank-content">
          <p>{product.rank_explanation.reason}</p>
          {product.rank_explanation.comparison && <p>{product.rank_explanation.comparison}</p>}
          {product.rank_explanation.budget && <p><strong>Budget</strong> {product.rank_explanation.budget}</p>}
          <p><strong>Availability</strong> {product.rank_explanation.availability}</p>
          {product.rank_explanation.caveats?.filter((note) => note !== product.rank_explanation.availability).map((note) => <p className="rank-caveat" key={note}>{note}</p>)}
        </div>
      </details> : <p className="match-line">{product.explanation || product.description || 'Product details available from the source.'}</p>}
      <div className="card-actions"><label className="compare-check"><input type="checkbox" checked={selected} onChange={() => onCompare(product)} /> Compare</label>{product.product_url ? <a href={product.product_url} target="_blank" rel="noopener noreferrer" className="view-link">View product</a> : <span className="unverified">Source link unavailable</span>}</div>
    </div>
  </article>
}

function RecommendationResults({ message, saved, selected, onSave, onCompare }) {
  const [sort, setSort] = useState('recommended')
  const products = [...message.products]
  if (sort === 'price_asc') products.sort((a, b) => (a.price ?? Infinity) - (b.price ?? Infinity))
  if (sort === 'price_desc') products.sort((a, b) => (b.price ?? -Infinity) - (a.price ?? -Infinity))
  return <div className="recommendation-results">
    {message.rankingExplanation?.summary && <p className="ranking-summary">{message.rankingExplanation.summary}</p>}
    <label className="result-sort">Display order
      <select value={sort} onChange={(event) => setSort(event.target.value)}>
        <option value="recommended">Recommended</option>
        <option value="price_asc">Price: low to high</option>
        <option value="price_desc">Price: high to low</option>
      </select>
    </label>
    <div className="recommendation-grid">{products.map((product) => <ProductCard key={product.product_id}
      product={product} saved={saved.some((item) => item.product_id === product.product_id)}
      selected={selected.some((item) => item.product_id === product.product_id)} onSave={onSave} onCompare={onCompare} />)}</div>
  </div>
}

function App() {
  const [initialChatState] = useState(readChatState)
  const [chatSessions, setChatSessions] = useState(initialChatState.chats)
  const [activeChatId, setActiveChatId] = useState(initialChatState.active.id)
  const [sessionId, setSessionId] = useState(initialChatState.active.id)
  const [messages, setMessages] = useState(initialChatState.active.messages || [])
  const [saved, setSaved] = useState(() => JSON.parse(localStorage.getItem(SAVED_KEY) || '[]'))
  const [selected, setSelected] = useState([])
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [providerStatus, setProviderStatus] = useState({})
  const [providerErrors, setProviderErrors] = useState({})
  const [view, setView] = useState('chat')
  const [mobileFilters, setMobileFilters] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    setChatSessions((items) => items.map((chat) => chat.id === activeChatId ? {
      ...chat,
      messages: messages.slice(-30),
      title: chat.title === 'New chat' ? (messages.find((message) => message.role === 'user')?.text?.slice(0, 42) || 'New chat') : chat.title,
      updatedAt: new Date().toISOString(),
    } : chat))
  }, [messages, activeChatId])
  useEffect(() => {
    localStorage.setItem(CHATS_KEY, JSON.stringify(chatSessions))
    localStorage.setItem(ACTIVE_CHAT_KEY, activeChatId)
  }, [chatSessions, activeChatId])
  useEffect(() => { localStorage.setItem(SAVED_KEY, JSON.stringify(saved)) }, [saved])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])

  function saveProduct(product) { setSaved((items) => items.some((item) => item.product_id === product.product_id) ? items.filter((item) => item.product_id !== product.product_id) : [...items, product]) }
  function compareProduct(product) { setSelected((items) => items.some((item) => item.product_id === product.product_id) ? items.filter((item) => item.product_id !== product.product_id) : items.length < 3 ? [...items, product] : items) }

  function startNewChat() {
    if (loading) return
    const chat = newChatRecord()
    setChatSessions((items) => [chat, ...items])
    setActiveChatId(chat.id)
    setSessionId(chat.id)
    setMessages([])
    setQuery('')
    setSelected([])
    setProviderStatus({})
    setProviderErrors({})
    setError('')
    setView('chat')
  }

  function openChat(chat) {
    if (chat.id === activeChatId) { setView('chat'); return }
    setActiveChatId(chat.id)
    setSessionId(chat.id)
    setMessages(chat.messages || [])
    setQuery('')
    setSelected([])
    setProviderStatus({})
    setProviderErrors({})
    setError('')
    setView('chat')
  }

  async function sendMessage(event, directQuery) {
    event?.preventDefault(); const text = (directQuery || query).trim(); if (!text || loading) return
    setQuery(''); setView('chat'); setError(''); setMessages((items) => [...items, { role: 'user', text, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]); setLoading(true)
    try {
      const response = await fetch(`${API_URL}/api/v1/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: sessionId, message: text }) })
      if (!response.ok) throw new Error(`Assistant request failed (${response.status})`)
      const data = await response.json()
      setProviderStatus(data.provider_status || {}); setProviderErrors(data.retrieval_errors || {})
      setMessages((items) => [...items, { role: 'assistant', text: data.message, products: data.products || [], rankingExplanation: data.ranking_explanation, action: data.action, constraints: data.active_constraints || {}, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }])
    } catch (requestError) { setMessages((items) => [...items, { role: 'assistant', text: requestError.message, error: true }]); setError(requestError.message) } finally { setLoading(false) }
  }

  const providerLabel = Object.values(providerStatus).some((status) => status === 'ready') ? 'Sources connected' : 'Search ready'
  const failedProviders = Object.keys(providerErrors).filter((name) => providerStatus[name] && providerStatus[name] !== 'ready')
  const unverifiedProviders = Object.keys(providerErrors).filter((name) => providerStatus[name] === 'ready')

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="logo"><span className="logo-mark"><Sparkles size={16} /></span><span>ShopSmart <b>AI</b></span></div>
      <button className="new-chat" onClick={startNewChat}><Plus size={17} /> New chat</button>
      <nav className="side-nav"><button className={view === 'chat' ? 'active' : ''} onClick={() => setView('chat')}><Bot size={17} /> Assistant</button><button className={view === 'searches' ? 'active' : ''} onClick={() => setView('searches')}><History size={17} /> My searches <span>{chatSessions.filter((chat) => chat.messages.length > 0).length || ''}</span></button><button className={view === 'saved' ? 'active' : ''} onClick={() => setView('saved')}><Bookmark size={17} /> Saved items <span>{saved.length || ''}</span></button><button className={view === 'compare' ? 'active' : ''} onClick={() => setView('compare')}><GitCompareArrows size={17} /> Compare <span>{selected.length || ''}</span></button></nav>
      <div className="chat-history" aria-label="Previous chats">{chatSessions.filter((chat) => chat.messages.length > 0).slice(0, 8).map((chat) => <button key={chat.id} className={`chat-history-item ${chat.id === activeChatId ? 'active' : ''}`} onClick={() => openChat(chat)}><span>{chat.title}</span><small>{new Date(chat.updatedAt).toLocaleDateString([], { month: 'short', day: 'numeric' })}</small></button>)}</div>
      <div className="sidebar-bottom"><div className="connection"><span className="status-dot" /> {providerLabel}</div><button><Settings size={17} /> Settings</button><button><CircleHelp size={17} /> Help center</button></div>
    </aside>
    <main className="workspace">
      <header className="topbar"><button className="mobile-only icon-button" onClick={() => setMobileFilters(!mobileFilters)} aria-label="Toggle filters"><Menu size={19} /></button><div><p className="kicker">ShopSmart AI / {view === 'chat' ? 'New discovery' : view}</p><h1>{view === 'chat' ? 'Your AI shopping assistant' : view === 'saved' ? 'Saved items' : view === 'compare' ? 'Compare products' : 'Your searches'}</h1></div><div className="top-status"><ShieldCheck size={16} /> Private by default</div></header>
      {view === 'searches' && <section className="searches-view"><div className="results-header"><div><p className="kicker">Conversation archive</p><h2>Your searches</h2></div><button className="compare-button" onClick={startNewChat}><Plus size={16} /> New chat</button></div><div className="search-list">{chatSessions.filter((chat) => chat.messages.length > 0).map((chat) => <button key={chat.id} onClick={() => openChat(chat)}><History size={16} /><span><b>{chat.title}</b><small>{chat.messages.length} messages · {new Date(chat.updatedAt).toLocaleString()}</small></span><ChevronDown size={15} /></button>)}</div></section>}
      {view === 'chat' && <section className="conversation"><div className="conversation-scroll">{messages.length === 0 && <div className="empty-state"><div className="empty-icon"><Sparkles size={25} /></div><h2>Find the right product, faster.</h2><p>Tell me what you need, your budget, and what matters most. I’ll search available sources and compare your options.</p><div className="suggestions">{suggestions.map((item) => <button key={item} onClick={() => sendMessage(null, item)}>{item}<span>↗</span></button>)}</div></div>}{messages.map((message, index) => <div className={`message-row ${message.role}`} key={`${index}-${message.text}`}><div className="message-avatar">{message.role === 'assistant' ? <Sparkles size={14} /> : 'You'}</div><div className="message-body"><div className={`message-bubble ${message.error ? 'error' : ''}`}>{message.text}</div><small>{message.time}</small>{message.products?.length > 0 && <RecommendationResults message={message} saved={saved} selected={selected} onSave={saveProduct} onCompare={compareProduct} />}{message.action === 'clarification' && <div className="follow-up"><button onClick={() => setQuery('Show me options under ₹5,000')}>Set a budget</button><button onClick={() => setQuery('Show me different brands')}>Different brands</button></div>}</div></div>)}{loading && <div className="search-progress"><Sparkles size={15} /> Understanding your requirements <span>Searching configured sources</span><i /></div>}<div ref={bottomRef} /></div><form className="message-form" onSubmit={sendMessage}><div className="input-wrap"><Sparkles size={18} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Describe what you’re looking for..." aria-label="Shopping request" /><span>Enter ↵</span></div><button disabled={loading || !query.trim()} aria-label="Send message"><Send size={17} /></button></form></section>}
          {failedProviders.length > 0 && <div className="provider-warning">Some sources were unavailable: {failedProviders.join(', ')}. Results shown are partial.</div>}{unverifiedProviders.length > 0 && <div className="provider-warning">Some {unverifiedProviders.join(', ')} pages were found, but their prices could not be verified for your budget.</div>}
    </main>
  </div>
}

function Comparison({ products, onRemove }) {
  if (products.length === 0) return <div className="empty-collection"><GitCompareArrows size={22} /><p>Select up to three products using Compare to see them side by side.</p></div>
  return <div className="comparison-grid">{products.map((product) => <article className="comparison-card" key={product.product_id}><button onClick={() => onRemove(product)} aria-label="Remove from comparison"><X size={15} /></button><h3>{product.name}</h3><span>{product.brand || product.category}</span><strong>{money(product.price)}</strong><dl><dt>Rating</dt><dd>{product.rating || 'Unavailable'}</dd><dt>Availability</dt><dd>{product.availability || (product.in_stock ? 'In stock' : 'Unavailable')}</dd><dt>Key details</dt><dd>{product.features?.slice(0, 3).join(', ') || 'Not provided'}</dd><dt>Source</dt><dd>{product.source_name || 'External source'}</dd></dl>{product.product_url && <a href={product.product_url} target="_blank" rel="noopener noreferrer">Open source</a>}</article>)}</div>
}

createRoot(document.getElementById('root')).render(<App />)