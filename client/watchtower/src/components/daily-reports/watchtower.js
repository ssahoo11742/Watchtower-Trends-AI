import React, { useState } from 'react';
import { Upload, ChevronDown, ChevronRight, TrendingUp, TrendingDown, Home, FileText, Settings, Search, BarChart3, Activity, Zap, Globe, Shield } from 'lucide-react';
import Papa from 'papaparse';

export default function WatchtowerApp() {
  const [data, setData] = useState([]);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [expandedCompany, setExpandedCompany] = useState(null);
  const [tradingStyle, setTradingStyle] = useState('swing');
  const [currentPage, setCurrentPage] = useState('home'); // 'home', 'daily-report', 'custom-jobs', 'ticker'
  const [selectedTicker, setSelectedTicker] = useState(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        setData(results.data);
        if (results.data.length > 0) {
          const firstTopic = results.data[0].Topic_Keywords;
          setSelectedTopic(firstTopic);
        }
      },
      error: (error) => {
        console.error('Error parsing CSV:', error);
      }
    });
  };

  const navigateTo = (page, ticker = null) => {
    setCurrentPage(page);
    if (ticker) setSelectedTicker(ticker);
    setExpandedCompany(null);
  };

  // Home Page
  const HomePage = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Hero Section */}
        <div className="text-center mb-16 pt-12">
          <div className="inline-flex items-center gap-3 mb-6">
            <Shield className="w-16 h-16 text-cyan-400" />
            <h1 className="text-7xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent">
              WATCHTOWER
            </h1>
          </div>
          <p className="text-2xl text-slate-300 mb-4">AI-Powered Stock Analysis Platform</p>
          <p className="text-lg text-slate-400 max-w-3xl mx-auto">
            Leveraging machine learning and real-time news analysis to identify trending industries and investment opportunities
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <button
            onClick={() => navigateTo('daily-report')}
            className="group bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700 hover:border-cyan-500 transition-all hover:shadow-2xl hover:shadow-cyan-500/20 text-left"
          >
            <div className="bg-gradient-to-br from-cyan-500 to-blue-600 w-16 h-16 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <FileText className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-2xl font-bold text-cyan-400 mb-3">Daily Report</h3>
            <p className="text-slate-400 mb-4">
              Curated daily analysis of trending stocks and industries based on latest news and ML insights
            </p>
            <div className="flex items-center text-cyan-400 font-medium">
              View Today's Report
              <ChevronRight className="w-5 h-5 ml-2 group-hover:translate-x-2 transition-transform" />
            </div>
          </button>

          <button
            onClick={() => navigateTo('custom-jobs')}
            className="group bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700 hover:border-purple-500 transition-all hover:shadow-2xl hover:shadow-purple-500/20 text-left"
          >
            <div className="bg-gradient-to-br from-purple-500 to-pink-600 w-16 h-16 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Settings className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-2xl font-bold text-purple-400 mb-3">Custom Jobs</h3>
            <p className="text-slate-400 mb-4">
              Configure and run personalized analysis jobs with custom date ranges, topics, and parameters
            </p>
            <div className="flex items-center text-purple-400 font-medium">
              Create Custom Job
              <ChevronRight className="w-5 h-5 ml-2 group-hover:translate-x-2 transition-transform" />
            </div>
          </button>

          <div className="group bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700 hover:border-blue-500 transition-all hover:shadow-2xl hover:shadow-blue-500/20">
            <div className="bg-gradient-to-br from-blue-500 to-indigo-600 w-16 h-16 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Search className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-2xl font-bold text-blue-400 mb-3">Ticker Search</h3>
            <p className="text-slate-400 mb-4">
              Deep dive into individual stocks with comprehensive charts, fundamentals, and technical analysis
            </p>
            <div className="flex items-center text-slate-500 font-medium">
              Coming Soon
            </div>
          </div>
        </div>

        {/* Stats Section */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-6 border border-slate-700">
            <div className="flex items-center gap-3 mb-2">
              <Activity className="w-6 h-6 text-emerald-400" />
              <h4 className="text-slate-400 text-sm">AI Models</h4>
            </div>
            <p className="text-3xl font-bold text-emerald-400">BERTopic</p>
            <p className="text-slate-500 text-sm mt-1">Topic modeling</p>
          </div>

          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-6 border border-slate-700">
            <div className="flex items-center gap-3 mb-2">
              <Globe className="w-6 h-6 text-blue-400" />
              <h4 className="text-slate-400 text-sm">Data Sources</h4>
            </div>
            <p className="text-3xl font-bold text-blue-400">NewsAPI</p>
            <p className="text-slate-500 text-sm mt-1">Real-time news</p>
          </div>

          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-6 border border-slate-700">
            <div className="flex items-center gap-3 mb-2">
              <BarChart3 className="w-6 h-6 text-purple-400" />
              <h4 className="text-slate-400 text-sm">Analysis</h4>
            </div>
            <p className="text-3xl font-bold text-purple-400">10K+</p>
            <p className="text-slate-500 text-sm mt-1">Companies tracked</p>
          </div>

          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-6 border border-slate-700">
            <div className="flex items-center gap-3 mb-2">
              <Zap className="w-6 h-6 text-amber-400" />
              <h4 className="text-slate-400 text-sm">Update Freq</h4>
            </div>
            <p className="text-3xl font-bold text-amber-400">Daily</p>
            <p className="text-slate-500 text-sm mt-1">Fresh insights</p>
          </div>
        </div>

        {/* Upload Section */}
        {data.length === 0 && (
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-12 text-center border border-slate-700">
            <Upload className="w-12 h-12 mx-auto mb-4 text-cyan-400" />
            <h2 className="text-2xl font-bold text-slate-200 mb-2">Upload Analysis Data</h2>
            <p className="text-slate-400 mb-6">Upload a CSV file to start exploring the daily report</p>
            <label className="inline-block cursor-pointer bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium px-6 py-3 rounded-lg transition-all shadow-lg hover:shadow-cyan-500/50">
              <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
              Upload CSV File
            </label>
          </div>
        )}

        {data.length > 0 && (
          <div className="bg-gradient-to-br from-cyan-900/20 to-blue-900/20 rounded-2xl p-8 text-center border border-cyan-700/50">
            <div className="flex items-center justify-center gap-3 mb-4">
              <div className="w-3 h-3 bg-emerald-400 rounded-full animate-pulse"></div>
              <h3 className="text-xl font-bold text-cyan-400">Data Loaded Successfully</h3>
            </div>
            <p className="text-slate-300 mb-6">
              {data.length} companies analyzed across {[...new Set(data.map(row => row.Topic_Keywords))].filter(Boolean).length} topic groups
            </p>
            <button
              onClick={() => navigateTo('daily-report')}
              className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium px-8 py-3 rounded-lg transition-all shadow-lg hover:shadow-cyan-500/50"
            >
              View Daily Report
            </button>
          </div>
        )}
      </div>
    </div>
  );

  // Custom Jobs Page (Placeholder)
  const CustomJobsPage = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      <div className="max-w-4xl mx-auto">
        <button
          onClick={() => navigateTo('home')}
          className="mb-6 flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors"
        >
          <Home className="w-5 h-5" />
          Back to Home
        </button>
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-12 text-center border border-slate-700">
          <Settings className="w-16 h-16 mx-auto mb-4 text-purple-400" />
          <h1 className="text-3xl font-bold text-purple-400 mb-4">Custom Jobs</h1>
          <p className="text-slate-400 text-lg">
            This feature is under development. Soon you'll be able to configure custom analysis jobs with your own parameters.
          </p>
        </div>
      </div>
    </div>
  );

  // Ticker Detail Page
  const TickerDetailPage = ({ ticker }) => {
    const tickerData = data.find(row => row.Ticker === ticker);
    
    if (!tickerData) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
          <div className="max-w-4xl mx-auto">
            <button
              onClick={() => navigateTo('daily-report')}
              className="mb-6 flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors"
            >
              <ChevronRight className="w-5 h-5 rotate-180" />
              Back to Daily Report
            </button>
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-12 text-center border border-slate-700">
              <p className="text-slate-400 text-lg">Ticker not found</p>
            </div>
          </div>
        </div>
      );
    }

    const getScoreColor = (score) => {
      if (!score || typeof score !== 'number') return 'text-slate-400';
      if (score >= 0.7) return 'text-emerald-400';
      if (score >= 0.5) return 'text-cyan-400';
      if (score >= 0.3) return 'text-amber-400';
      return 'text-red-400';
    };

    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
        <div className="max-w-7xl mx-auto">
          <button
            onClick={() => navigateTo('daily-report')}
            className="mb-6 flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors"
          >
            <ChevronRight className="w-5 h-5 rotate-180" />
            Back to Daily Report
          </button>

          {/* Header */}
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700 mb-6 shadow-xl">
            <div className="flex items-start justify-between">
              <div>
                <h1 className="text-4xl font-bold text-cyan-400 mb-2">{tickerData.Ticker}</h1>
                <p className="text-xl text-slate-300 mb-4">{tickerData.Company_Name}</p>
                <div className="flex items-center gap-4">
                  <div>
                    <span className="text-3xl font-bold text-slate-200">
                      ${typeof tickerData.Current_Price === 'number' ? tickerData.Current_Price.toFixed(2) : 'N/A'}
                    </span>
                  </div>
                  {typeof tickerData.Change_1D === 'number' && (
                    <div className={`flex items-center gap-2 text-xl font-semibold ${tickerData.Change_1D >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {tickerData.Change_1D >= 0 ? <TrendingUp className="w-6 h-6" /> : <TrendingDown className="w-6 h-6" />}
                      {tickerData.Change_1D > 0 ? '+' : ''}{tickerData.Change_1D.toFixed(2)}%
                    </div>
                  )}
                </div>
              </div>
              <div className="text-right">
                <div className="text-slate-400 text-sm mb-1">Relevance Score</div>
                <div className={`text-3xl font-bold ${getScoreColor(tickerData.Relevance_Score)}`}>
                  {typeof tickerData.Relevance_Score === 'number' ? tickerData.Relevance_Score.toFixed(3) : 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* TradingView Chart */}
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-6 border border-slate-700 mb-6 shadow-xl">
            <h3 className="font-semibold text-cyan-400 mb-4 text-xl">Price Chart</h3>
            <div className="tradingview-widget-container" style={{ height: '500px', width: '100%' }}>
              <iframe
                scrolling="no"
                allowTransparency={true}
                frameBorder="0"
                src={`https://s.tradingview.com/widgetembed/?frameElementId=tradingview_chart&symbol=${tickerData.Ticker}&interval=D&hidesidetoolbar=0&symboledit=1&saveimage=1&toolbarbg=1e293b&studies=[]&theme=dark&style=1&timezone=Etc%2FUTC&withdateranges=1&studies_overrides={}&overrides={}&enabled_features=[]&disabled_features=[]&locale=en&utm_source=localhost&utm_medium=widget_new&utm_campaign=chart&utm_term=${tickerData.Ticker}`}
                style={{ width: '100%', height: '100%', margin: 0, padding: 0 }}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Trading Analysis */}
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-6 border border-slate-700 shadow-xl">
              <h3 className="font-semibold text-cyan-400 mb-4 text-xl">Trading Analysis</h3>
              <div className="space-y-3">
                {[
                  { label: 'Day Trading', score: tickerData.Day_Trade_Score, rating: tickerData.Day_Trade_Rating },
                  { label: 'Swing Trading', score: tickerData.Swing_Trade_Score, rating: tickerData.Swing_Trade_Rating },
                  { label: 'Position Trading', score: tickerData.Position_Trade_Score, rating: tickerData.Position_Trade_Rating },
                  { label: 'Long-Term', score: tickerData.LongTerm_Score, rating: tickerData.LongTerm_Rating }
                ].map(({ label, score, rating }) => (
                  <div key={label} className="flex justify-between items-center p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                    <span className="text-slate-300 font-medium">{label}</span>
                    <div className="text-right">
                      <div className={`text-xl font-bold ${getScoreColor(score)}`}>
                        {typeof score === 'number' ? score.toFixed(3) : 'N/A'}
                      </div>
                      <div className="text-xs text-slate-400">{rating || 'N/A'}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Price Performance */}
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-6 border border-slate-700 shadow-xl">
              <h3 className="font-semibold text-cyan-400 mb-4 text-xl">Price Performance</h3>
              <div className="space-y-3">
                {[
                  { label: '1 Day', value: tickerData.Change_1D },
                  { label: '1 Week', value: tickerData.Change_1W },
                  { label: '1 Month', value: tickerData.Change_1M },
                  { label: '3 Months', value: tickerData.Change_3M },
                  { label: '1 Year', value: tickerData.Change_1Y }
                ].map(({ label, value }) => (
                  <div key={label} className="flex justify-between items-center p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                    <span className="text-slate-300 font-medium">{label}</span>
                    {typeof value === 'number' ? (
                      <span className={`font-semibold text-lg ${value >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {value > 0 ? '+' : ''}{value.toFixed(2)}%
                      </span>
                    ) : (
                      <span className="text-slate-500">—</span>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Technical Indicators */}
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-6 border border-slate-700 shadow-xl">
              <h3 className="font-semibold text-cyan-400 mb-4 text-xl">Technical Indicators</h3>
              <div className="space-y-3">
                <div className="flex justify-between p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                  <span className="text-slate-300 font-medium">RSI (14)</span>
                  <span className="font-semibold text-slate-200">{tickerData.RSI_14 || 'N/A'}</span>
                </div>
                <div className="flex justify-between p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                  <span className="text-slate-300 font-medium">Price vs MA50</span>
                  <span className="font-semibold text-slate-200">
                    {typeof tickerData.Price_vs_MA50 === 'number' ? `${tickerData.Price_vs_MA50.toFixed(2)}%` : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                  <span className="text-slate-300 font-medium">Price vs MA200</span>
                  <span className="font-semibold text-slate-200">
                    {typeof tickerData.Price_vs_MA200 === 'number' ? `${tickerData.Price_vs_MA200.toFixed(2)}%` : 'N/A'}
                  </span>
                </div>
              </div>
            </div>

            {/* Fundamentals */}
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-6 border border-slate-700 shadow-xl">
              <h3 className="font-semibold text-cyan-400 mb-4 text-xl">Fundamentals</h3>
              <div className="space-y-3">
                <div className="flex justify-between p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                  <span className="text-slate-300 font-medium">P/E Ratio</span>
                  <span className="font-semibold text-slate-200">{tickerData.PE_Ratio || 'N/A'}</span>
                </div>
                <div className="flex justify-between p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                  <span className="text-slate-300 font-medium">Dividend Yield</span>
                  <span className="font-semibold text-slate-200">
                    {tickerData.Dividend_Yield ? `${tickerData.Dividend_Yield}%` : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                  <span className="text-slate-300 font-medium">Profit Margin</span>
                  <span className="font-semibold text-slate-200">
                    {typeof tickerData.Profit_Margin === 'number' ? `${tickerData.Profit_Margin.toFixed(2)}%` : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                  <span className="text-slate-300 font-medium">ROE</span>
                  <span className="font-semibold text-slate-200">
                    {typeof tickerData.ROE === 'number' ? `${tickerData.ROE.toFixed(2)}%` : 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Daily Report Page
  const DailyReportPage = () => {
    const topics = [...new Set(data.map(row => row.Topic_Keywords))].filter(Boolean);
    
    let filteredData = selectedTopic 
      ? data.filter(row => row.Topic_Keywords === selectedTopic)
      : data;

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

    const getScoreColor = (score) => {
      if (!score || typeof score !== 'number') return 'text-slate-400';
      if (score >= 0.7) return 'text-emerald-400';
      if (score >= 0.5) return 'text-cyan-400';
      if (score >= 0.3) return 'text-amber-400';
      return 'text-red-400';
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

    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <div className="flex h-screen">
          {/* Sidebar */}
          <div className="w-80 bg-slate-900 border-r border-slate-700 overflow-y-auto shadow-xl">
            <div className="p-6 border-b border-slate-700 bg-gradient-to-r from-slate-900 to-slate-800">
              <button
                onClick={() => navigateTo('home')}
                className="mb-4 flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors text-sm"
              >
                <Home className="w-4 h-4" />
                Home
              </button>
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
                  <button
                    onClick={() => setTradingStyle('relevance')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      tradingStyle === 'relevance' 
                        ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-900/50' 
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
                    }`}
                  >
                    Relevance
                  </button>
                  <button
                    onClick={() => setTradingStyle('day')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      tradingStyle === 'day' 
                        ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-900/50' 
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
                    }`}
                  >
                    Day
                  </button>
                  <button
                    onClick={() => setTradingStyle('swing')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      tradingStyle === 'swing' 
                        ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-900/50' 
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
                    }`}
                  >
                    Swing
                  </button>
                  <button
                    onClick={() => setTradingStyle('position')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      tradingStyle === 'position' 
                        ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-900/50' 
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
                    }`}
                  >
                    Position
                  </button>
                  <button
                    onClick={() => setTradingStyle('long')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      tradingStyle === 'long' 
                        ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-900/50' 
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
                    }`}
                  >
                    Long-Term
                  </button>
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
                              </div>
                            </div>
                            
                            <div className="text-center">
                              <div className={`text-lg font-bold ${getScoreColor(tradingScore)}`}>
                                {typeof tradingScore === 'number' ? tradingScore.toFixed(3) : 'N/A'}
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
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                navigateTo('ticker', ticker);
                              }}
                              className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium px-6 py-3 rounded-lg transition-all shadow-lg hover:shadow-cyan-500/50 flex items-center gap-2"
                            >
                              View Full Details
                              <ChevronRight className="w-5 h-5" />
                            </button>
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
  };

  // Route to appropriate page
  if (data.length === 0) {
    return <HomePage />;
  }

  switch (currentPage) {
    case 'home':
      return <HomePage />;
    case 'daily-report':
      return <DailyReportPage />;
    case 'custom-jobs':
      return <CustomJobsPage />;
    case 'ticker':
      return <TickerDetailPage ticker={selectedTicker} />;
    default:
      return <HomePage />;
  }
}