// src/pages/DailyReportPage.jsx - Responsive (Desktop + Mobile)
import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Upload, ChevronDown, ChevronRight, TrendingUp, TrendingDown, Home, Menu, X, Filter } from 'lucide-react';
import { useData } from '../../contexts/DataContext';

export const DailyReportPage = () => {
  const { data, handleFileUpload } = useData();
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [expandedCompany, setExpandedCompany] = useState(null);
  const [tradingStyle, setTradingStyle] = useState('swing');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);

  useEffect(() => {
    if (data.length > 0 && !selectedTopic) {
      const firstTopic = data[0].Topic_Keywords;
      setSelectedTopic(firstTopic);
    }
  }, [data, selectedTopic]);

  const getScoreColor = (score) => {
    if (!score || typeof score !== 'number') return 'text-slate-400';
    if (score >= 0.7) return 'text-emerald-400';
    if (score >= 0.5) return 'text-cyan-400';
    if (score >= 0.3) return 'text-amber-400';
    return 'text-red-400';
  };

  const getSignalStyle = (signal) => {
    if (!signal) return { bg: 'bg-slate-700', text: 'text-slate-300', border: 'border-slate-600' };
    const str = String(signal).toUpperCase();
    if (str.includes('STRONG BUY')) return { bg: 'bg-emerald-900', text: 'text-emerald-300', border: 'border-emerald-700' };
    if (str.includes('BUY')) return { bg: 'bg-green-900', text: 'text-green-300', border: 'border-green-700' };
    if (str.includes('BULLISH') || str.includes('UPTREND')) return { bg: 'bg-blue-900', text: 'text-blue-300', border: 'border-blue-700' };
    if (str.includes('HOLD') || str.includes('CONSOLIDATING')) return { bg: 'bg-amber-900', text: 'text-amber-300', border: 'border-amber-700' };
    if (str.includes('SELL') || str.includes('AVOID') || str.includes('DOWNTREND')) return { bg: 'bg-red-900', text: 'text-red-300', border: 'border-red-700' };
    if (str.includes('CAUTION') || str.includes('MIXED')) return { bg: 'bg-orange-900', text: 'text-orange-300', border: 'border-orange-700' };
    return { bg: 'bg-slate-700', text: 'text-slate-300', border: 'border-slate-600' };
  };

  const getCurrentScore = (row) => {
    switch(tradingStyle) {
      case 'day': return row.Day_Trade_Score;
      case 'swing': return row.Swing_Trade_Score;
      case 'position': return row.Position_Trade_Score;
      case 'long': return row.LongTerm_Score;
      case 'relevance': return row.Swing_Trade_Score;
      default: return row.Swing_Trade_Score;
    }
  };

  const getCurrentRating = (row) => {
    switch(tradingStyle) {
      case 'day': return row.Day_Trade_Rating;
      case 'swing': return row.Swing_Trade_Rating;
      case 'position': return row.Position_Trade_Rating;
      case 'long': return row.LongTerm_Rating;
      case 'relevance': return row.Swing_Trade_Rating;
      default: return row.Swing_Trade_Rating;
    }
  };

  if (data.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4 md:p-8">
        <div className="max-w-xl mx-auto md:mt-8">
          <Link
            to="/"
            className="mb-6 flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors"
          >
            <Home className="w-5 h-5" />
            Back to Home
          </Link>
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl shadow-2xl p-8 md:p-12 text-center border border-slate-700">
            <div className="mb-6">
              <div className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-2">
                WATCHTOWER
              </div>
              <div className="text-slate-400 text-xs md:text-sm tracking-wider">STOCK ANALYSIS PLATFORM</div>
            </div>
            <Upload className="w-12 h-12 mx-auto mb-4 text-cyan-400" />
            <h1 className="text-xl md:text-2xl font-bold text-slate-200 mb-2">Upload Data</h1>
            <p className="text-slate-400 mb-6 text-sm">Upload your CSV file to begin analysis</p>
            <label className="inline-block cursor-pointer bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium px-6 py-3 rounded-lg transition-all shadow-lg hover:shadow-cyan-500/50">
              <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
              Upload CSV File
            </label>
          </div>
        </div>
      </div>
    );
  }

  const topics = [...new Set(data.map(row => row.Topic_Keywords))].filter(Boolean);
  
  let filteredData = selectedTopic 
    ? data.filter(row => row.Topic_Keywords === selectedTopic)
    : data;

  filteredData = [...filteredData].sort((a, b) => {
    let scoreA, scoreB;
    
    if (tradingStyle === 'relevance') {
      scoreA = a.Relevance_Score;
      scoreB = b.Relevance_Score;
    } else {
      scoreA = getCurrentScore(a);
      scoreB = getCurrentScore(b);
    }
    
    if (scoreA == null && scoreB == null) return 0;
    if (scoreA == null) return 1;
    if (scoreB == null) return -1;
    
    return scoreB - scoreA;
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* MOBILE LAYOUT */}
      <div className="md:hidden">
        {/* Mobile Header */}
        <div className="sticky top-0 z-50 bg-gradient-to-r from-slate-900 to-slate-800 border-b border-slate-700 shadow-xl">
          <div className="flex items-center justify-between p-4">
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <Menu className="w-6 h-6 text-cyan-400" />
            </button>
            
            <div className="flex-1 text-center">
              <div className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                WATCHTOWER
              </div>
              <div className="text-xs text-slate-400">{filteredData.length} companies</div>
            </div>
            
            <button
              onClick={() => setFilterOpen(!filterOpen)}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <Filter className="w-6 h-6 text-cyan-400" />
            </button>
          </div>

          {/* Trading Style Filter - Collapsible */}
          {filterOpen && (
            <div className="px-4 pb-4 space-y-2">
              <div className="text-xs text-slate-400 mb-2">Trading Style</div>
              <div className="grid grid-cols-3 gap-2">
                {['relevance', 'day', 'swing'].map((style) => (
                  <button
                    key={style}
                    onClick={() => {
                      setTradingStyle(style);
                      setFilterOpen(false);
                    }}
                    className={`px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                      tradingStyle === style
                        ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg' 
                        : 'bg-slate-800 text-slate-300 border border-slate-700'
                    }`}
                  >
                    {style.charAt(0).toUpperCase() + style.slice(1)}
                  </button>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-2">
                {['position', 'long'].map((style) => (
                  <button
                    key={style}
                    onClick={() => {
                      setTradingStyle(style);
                      setFilterOpen(false);
                    }}
                    className={`px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                      tradingStyle === style
                        ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg' 
                        : 'bg-slate-800 text-slate-300 border border-slate-700'
                    }`}
                  >
                    {style === 'long' ? 'Long-Term' : style.charAt(0).toUpperCase() + style.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Selected Topic Display */}
          <div className="px-4 pb-3">
            <div className="text-sm font-semibold text-cyan-400 truncate">{selectedTopic}</div>
          </div>
        </div>

        {/* Sidebar Overlay */}
        {sidebarOpen && (
          <div 
            className="fixed inset-0 bg-black/50 z-50"
            onClick={() => setSidebarOpen(false)}
          >
            <div 
              className="w-80 max-w-[85vw] h-full bg-slate-900 shadow-2xl overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-4 border-b border-slate-700 flex items-center justify-between sticky top-0 bg-slate-900 z-10">
                <div>
                  <div className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                    WATCHTOWER
                  </div>
                  <p className="text-xs text-slate-400">{topics.length} Topics</p>
                </div>
                <button
                  onClick={() => setSidebarOpen(false)}
                  className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-slate-400" />
                </button>
              </div>
              
              <div className="p-3">
                <Link
                  to="/"
                  className="mb-3 flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors text-sm px-3 py-2"
                  onClick={() => setSidebarOpen(false)}
                >
                  <Home className="w-4 h-4" />
                  Home
                </Link>
                
                {topics.map((topic, idx) => {
                  const count = data.filter(row => row.Topic_Keywords === topic).length;
                  return (
                    <button
                      key={idx}
                      onClick={() => {
                        setSelectedTopic(topic);
                        setExpandedCompany(null);
                        setSidebarOpen(false);
                      }}
                      className={`w-full text-left px-3 py-3 rounded-lg mb-2 transition-all ${
                        selectedTopic === topic
                          ? 'bg-gradient-to-r from-cyan-900/50 to-blue-900/50 text-cyan-300 font-medium border border-cyan-700'
                          : 'hover:bg-slate-800 text-slate-300 border border-transparent'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <span className="text-xs leading-tight pr-2">{topic}</span>
                        <span className={`text-xs px-2 py-1 rounded-full flex-shrink-0 ${
                          selectedTopic === topic 
                            ? 'bg-cyan-700 text-cyan-200' 
                            : 'bg-slate-700 text-slate-300'
                        }`}>
                          {count}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
              
              <div className="p-4 border-t border-slate-700 sticky bottom-0 bg-slate-900">
                <label className="block cursor-pointer bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium px-4 py-2 rounded-lg transition-colors text-center text-sm border border-slate-700">
                  <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
                  Upload New File
                </label>
              </div>
            </div>
          </div>
        )}

        {/* Main Content - Mobile Cards */}
        <div className="p-3 space-y-3 pb-20">
          {filteredData.map((row, idx) => {
            const ticker = row.Ticker;
            const name = row.Company_Name;
            const relevanceScore = row.Relevance_Score;
            const tradingScore = getCurrentScore(row);
            const rating = getCurrentRating(row);
            const price = row.Current_Price;
            const change1d = row.Change_1D;
            const mentions = row.Mentions;
            const isExpanded = expandedCompany === row;
            const ratingStyle = getSignalStyle(rating);

            return (
              <div 
                key={idx} 
                className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700 overflow-hidden"
              >
                <button
                  onClick={() => setExpandedCompany(isExpanded ? null : row)}
                  className="w-full p-4 hover:bg-slate-800/50 transition-colors"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <div className="text-lg font-bold text-cyan-300">{ticker}</div>
                        {isExpanded ? (
                          <ChevronDown className="w-5 h-5 text-cyan-400 flex-shrink-0" />
                        ) : (
                          <ChevronRight className="w-5 h-5 text-slate-500 flex-shrink-0" />
                        )}
                      </div>
                      <div className="text-xs text-slate-400 line-clamp-1">{name}</div>
                    </div>
                    
                    <div className="text-right ml-2">
                      <div className="text-base font-semibold text-slate-200">
                        ${typeof price === 'number' ? price.toFixed(2) : 'N/A'}
                      </div>
                      {typeof change1d === 'number' && (
                        <div className={`text-xs font-medium flex items-center justify-end gap-1 ${
                          change1d >= 0 ? 'text-emerald-400' : 'text-red-400'
                        }`}>
                          {change1d >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                          {change1d > 0 ? '+' : ''}{change1d.toFixed(2)}%
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div>
                      <div className="text-xs text-slate-500 mb-1">Relevance</div>
                      <div className={`text-sm font-bold ${getScoreColor(relevanceScore)}`}>
                        {typeof relevanceScore === 'number' ? relevanceScore.toFixed(3) : 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-1">{tradingStyle === 'relevance' ? 'Swing' : tradingStyle}</div>
                      <div className={`text-sm font-bold ${getScoreColor(tradingScore)}`}>
                        {typeof tradingScore === 'number' ? tradingScore.toFixed(3) : 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 mb-1">Mentions</div>
                      <div className="text-sm font-semibold text-slate-300">{mentions || 0}</div>
                    </div>
                  </div>

                  {rating && (
                    <div className="mt-3 flex justify-center">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium border ${ratingStyle.bg} ${ratingStyle.text} ${ratingStyle.border}`}>
                        {rating}
                      </span>
                    </div>
                  )}
                </button>

                {isExpanded && (
                  <div className="border-t border-slate-700 bg-slate-900/50 p-4">
                    <div className="space-y-4">
                      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg p-3 border border-slate-700">
                        <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Quick Stats</h4>
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-slate-400">P/E Ratio</span>
                            <span className="text-slate-200 font-medium">{row.PE_Ratio || 'N/A'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Dividend Yield</span>
                            <span className="text-slate-200 font-medium">
                              {row.Dividend_Yield ? `${row.Dividend_Yield}%` : 'N/A'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">RSI (14)</span>
                            <span className="text-slate-200 font-medium">{row.RSI_14 || 'N/A'}</span>
                          </div>
                        </div>
                      </div>

                      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg p-3 border border-slate-700">
                        <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Recent Performance</h4>
                        <div className="space-y-2 text-xs">
                          {[
                            { label: '1 Week', value: row.Change_1W },
                            { label: '1 Month', value: row.Change_1M },
                            { label: '3 Months', value: row.Change_3M }
                          ].map(({ label, value }) => (
                            <div key={label} className="flex justify-between">
                              <span className="text-slate-400">{label}</span>
                              {typeof value === 'number' ? (
                                <span className={`font-medium ${value >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                  {value > 0 ? '+' : ''}{value.toFixed(2)}%
                                </span>
                              ) : (
                                <span className="text-slate-500">—</span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg p-3 border border-slate-700">
                        <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Trading Scores</h4>
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-slate-400">Day</span>
                            <span className={`font-medium ${getScoreColor(row.Day_Trade_Score)}`}>
                              {typeof row.Day_Trade_Score === 'number' ? row.Day_Trade_Score.toFixed(3) : 'N/A'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Swing</span>
                            <span className={`font-medium ${getScoreColor(row.Swing_Trade_Score)}`}>
                              {typeof row.Swing_Trade_Score === 'number' ? row.Swing_Trade_Score.toFixed(3) : 'N/A'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Long-Term</span>
                            <span className={`font-medium ${getScoreColor(row.LongTerm_Score)}`}>
                              {typeof row.LongTerm_Score === 'number' ? row.LongTerm_Score.toFixed(3) : 'N/A'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <Link
                      to={`/ticker/${ticker}?from=daily-report`}
                      className="mt-4 block bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium px-4 py-3 rounded-lg transition-all shadow-lg text-center text-sm"
                    >
                      View Full Details →
                    </Link>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* DESKTOP LAYOUT */}
      <div className="hidden md:flex h-screen">
        {/* Sidebar */}
        <div className="w-80 bg-slate-900 border-r border-slate-700 overflow-y-auto shadow-xl">
          <div className="p-6 border-b border-slate-700 bg-gradient-to-r from-slate-900 to-slate-800">
            <Link
              to="/"
              className="mb-4 flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors text-sm"
            >
              <Home className="w-4 h-4" />
              Home
            </Link>
            <div className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-1">
              WATCHTOWER
            </div>
            <p className="text-sm text-slate-400">{topics.length} Topic Groups</p>
          </div>
          <div className="p-3">
            {topics.map((topic, idx) => {
              const count = data.filter(row => row.Topic_Keywords === topic).length;
              return (
                <button
                  key={idx}
                  onClick={() => {
                    setSelectedTopic(topic);
                    setExpandedCompany(null);
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg mb-2 transition-all ${
                    selectedTopic === topic
                      ? 'bg-gradient-to-r from-cyan-900/50 to-blue-900/50 text-cyan-300 font-medium border border-cyan-700 shadow-lg shadow-cyan-900/50'
                      : 'hover:bg-slate-800 text-slate-300 border border-transparent'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <span className="text-sm leading-tight">{topic}</span>
                    <span className={`text-xs px-2 py-1 rounded-full ml-2 flex-shrink-0 ${
                      selectedTopic === topic 
                        ? 'bg-cyan-700 text-cyan-200' 
                        : 'bg-slate-700 text-slate-300'
                    }`}>
                      {count}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
          <div className="p-6 border-t border-slate-700">
            <label className="block cursor-pointer bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium px-4 py-2 rounded-lg transition-colors text-center text-sm border border-slate-700">
              <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
              Upload New File
            </label>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="bg-gradient-to-r from-slate-900 to-slate-800 border-b border-slate-700 p-6 shadow-xl">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h2 className="text-2xl font-bold text-cyan-400">{selectedTopic}</h2>
                <p className="text-slate-400">{filteredData.length} companies</p>
              </div>
              <div className="flex gap-2">
                {['relevance', 'day', 'swing', 'position', 'long'].map((style) => (
                  <button
                    key={style}
                    onClick={() => setTradingStyle(style)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      tradingStyle === style
                        ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-900/50' 
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
                    }`}
                  >
                    {style === 'long' ? 'Long-Term' : style.charAt(0).toUpperCase() + style.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Column Headers */}
            <div className="grid grid-cols-7 gap-4 px-6 py-3 bg-slate-800/50 rounded-lg text-xs font-semibold text-cyan-400 uppercase border border-slate-700">
              <div>Company</div>
              <div className="text-center">Relevance</div>
              <div className="text-center">{tradingStyle === 'relevance' ? 'Swing Score' : `${tradingStyle} Score`}</div>
              <div className="text-center">Rating</div>
              <div className="text-center">Price</div>
              <div className="text-center">1D Change</div>
              <div className="text-center">Mentions</div>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-6 bg-slate-900">
            <div className="space-y-3">
              {filteredData.map((row, idx) => {
                const ticker = row.Ticker;
                const name = row.Company_Name;
                const relevanceScore = row.Relevance_Score;
                const tradingScore = getCurrentScore(row);
                const rating = getCurrentRating(row);
                const price = row.Current_Price;
                const change1d = row.Change_1D;
                const mentions = row.Mentions;
                const isExpanded = expandedCompany === row;
                const ratingStyle = getSignalStyle(rating);

                return (
                  <div key={idx} className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg border border-slate-700 overflow-hidden hover:shadow-xl hover:shadow-cyan-900/20 transition-all hover:border-cyan-700/50">
                    <button
                      onClick={() => setExpandedCompany(isExpanded ? null : row)}
                      className="w-full px-6 py-4 hover:bg-slate-800/50 transition-colors"
                    >
                      <div className="flex items-center gap-4">
                        {isExpanded ? (
                          <ChevronDown className="w-5 h-5 text-cyan-400 flex-shrink-0" />
                        ) : (
                          <ChevronRight className="w-5 h-5 text-slate-500 flex-shrink-0" />
                        )}
                        
                        <div className="flex-1 grid grid-cols-7 gap-4 items-center text-left">
                          <div>
                            <div className="font-bold text-cyan-300">{ticker}</div>
                            <div className="text-sm text-slate-400 truncate">{name}</div>
                          </div>
                          
                          <div className="text-center">
                            <div className={`text-lg font-bold ${getScoreColor(relevanceScore)}`}>
                              {typeof relevanceScore === 'number' ? relevanceScore.toFixed(3) : 'N/A'}
                              {console.log('relevanceScore', relevanceScore)}
                            </div>
                          </div>

                          <div className="text-center">
                            <div className={`text-lg font-bold ${getScoreColor(tradingScore)}`}>
                              {typeof tradingScore === 'number' ? tradingScore.toFixed(3) : 'N/A'}
                              {console.log('tradingScore', tradingScore)}
                            </div>
                          </div>
                          
                          <div className="flex justify-center">
                            {rating ? (
                              <span className={`px-3 py-1 rounded-full text-xs font-medium border ${ratingStyle.bg} ${ratingStyle.text} ${ratingStyle.border}`}>
                                {rating}
                              </span>
                            ) : (
                              <span className="text-slate-500 text-sm">—</span>
                            )}
                          </div>
                          
                          <div className="text-center">
                            <div className="font-semibold text-slate-200">
                              ${typeof price === 'number' ? price.toFixed(2) : 'N/A'}
                            </div>
                          </div>
                          
                          <div className="text-center">
                            {typeof change1d === 'number' ? (
                              <div className={`flex items-center justify-center gap-1 font-semibold ${change1d >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                {change1d >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                                {change1d > 0 ? '+' : ''}{change1d.toFixed(2)}%
                              </div>
                            ) : (
                              <span className="text-slate-500">—</span>
                            )}
                          </div>
                          
                          <div className="text-center">
                            <span className="text-slate-300 font-medium">{mentions || 0}</span>
                          </div>
                        </div>
                      </div>
                    </button>

                    {isExpanded && (
                      <div className="border-t border-slate-700 bg-slate-900/50 p-6">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                          {/* Quick Stats */}
                          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg p-4 border border-slate-700">
                            <h4 className="text-cyan-400 font-semibold mb-3">Quick Stats</h4>
                            <div className="space-y-2 text-sm">
                              <div className="flex justify-between">
                                <span className="text-slate-400">P/E Ratio</span>
                                <span className="text-slate-200 font-medium">{row.PE_Ratio || 'N/A'}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-slate-400">Dividend Yield</span>
                                <span className="text-slate-200 font-medium">
                                  {row.Dividend_Yield ? `${row.Dividend_Yield}%` : 'N/A'}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-slate-400">RSI (14)</span>
                                <span className="text-slate-200 font-medium">{row.RSI_14 || 'N/A'}</span>
                              </div>
                            </div>
                          </div>

                          {/* Recent Performance */}
                          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg p-4 border border-slate-700">
                            <h4 className="text-cyan-400 font-semibold mb-3">Recent Performance</h4>
                            <div className="space-y-2 text-sm">
                              {[
                                { label: '1 Week', value: row.Change_1W },
                                { label: '1 Month', value: row.Change_1M },
                                { label: '3 Months', value: row.Change_3M }
                              ].map(({ label, value }) => (
                                <div key={label} className="flex justify-between">
                                  <span className="text-slate-400">{label}</span>
                                  {typeof value === 'number' ? (
                                    <span className={`font-medium ${value >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                      {value > 0 ? '+' : ''}{value.toFixed(2)}%
                                    </span>
                                  ) : (
                                    <span className="text-slate-500">—</span>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Trading Scores */}
                          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg p-4 border border-slate-700">
                            <h4 className="text-cyan-400 font-semibold mb-3">Trading Scores</h4>
                            <div className="space-y-2 text-sm">
                              <div className="flex justify-between">
                                <span className="text-slate-400">Day</span>
                                <span className={`font-medium ${getScoreColor(row.Day_Trade_Score)}`}>
                                  {typeof row.Day_Trade_Score === 'number' ? row.Day_Trade_Score.toFixed(3) : 'N/A'}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-slate-400">Swing</span>
                                <span className={`font-medium ${getScoreColor(row.Swing_Trade_Score)}`}>
                                  {typeof row.Swing_Trade_Score === 'number' ? row.Swing_Trade_Score.toFixed(3) : 'N/A'}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-slate-400">Long-Term</span>
                                <span className={`font-medium ${getScoreColor(row.LongTerm_Score)}`}>
                                  {typeof row.LongTerm_Score === 'number' ? row.LongTerm_Score.toFixed(3) : 'N/A'}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="flex justify-center">
                          <Link
                            to={`/ticker/${ticker}?from=daily-report`}
                            className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium px-6 py-3 rounded-lg transition-all shadow-lg hover:shadow-cyan-500/50 flex items-center gap-2"
                          >
                            View Full Details
                            <ChevronRight className="w-5 h-5" />
                          </Link>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}