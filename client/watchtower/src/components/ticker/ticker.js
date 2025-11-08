// src/pages/TickerDetailPage.jsx
import { useParams, Link, useNavigate } from 'react-router-dom';
import { ChevronRight, TrendingUp, TrendingDown, Search, X } from 'lucide-react';
import { useData } from '../../contexts/DataContext';
import { useState, useEffect } from 'react';
import Papa from 'papaparse';
import { useSearchParams } from 'react-router-dom';

export const TickerDetailPage = () => {
  const { ticker } = useParams();
  const navigate = useNavigate();
  const { data } = useData();
  const [searchParams] = useSearchParams(); // Gets query params
  const from = searchParams.get('from'); // Gets value from ?from=home

  // Search page state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  
  // Check if companies data is already in sessionStorage
  const getCompaniesData = () => {
    const cached = sessionStorage.getItem('companiesData');
    return cached ? JSON.parse(cached) : null;
  };
  
  const [companiesData, setCompaniesData] = useState(getCompaniesData() || []);
  const [companiesLoaded, setCompaniesLoaded] = useState(!!getCompaniesData());
  
  const contextTickerData = data.find(row => row.Ticker === ticker);
  
  // State for fetched data and loading
  const [tickerData, setTickerData] = useState(contextTickerData || null);
  const [loading, setLoading] = useState(!contextTickerData && ticker !== '__srch__');
  const [error, setError] = useState(null);

  // If ticker is __srch__, show search page
  const isSearchPage = ticker === '__srch__';

  // Load companies.csv on mount (only if not already cached)
  useEffect(() => {
    // Skip if already loaded from sessionStorage
    if (companiesLoaded && companiesData.length > 0) {
      console.log('Companies loaded from cache:', companiesData.length);
      return;
    }

    const loadCompanies = async () => {
      try {
        const response = await fetch('/companies.csv');
        const csvText = await response.text();
        
        Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            setCompaniesData(results.data);
            setCompaniesLoaded(true);
            // Cache in sessionStorage for the session
            sessionStorage.setItem('companiesData', JSON.stringify(results.data));
            console.log('Loaded and cached companies:', results.data.length);
          },
          error: (error) => {
            console.error('Error parsing companies CSV:', error);
            setCompaniesLoaded(true);
          }
        });
      } catch (error) {
        console.error('Error loading companies CSV:', error);
        setCompaniesLoaded(true);
      }
    };

    loadCompanies();
  }, []);

  useEffect(() => {
    if (isSearchPage) return;

    const fetchTickerData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`http://localhost:8000/api/ticker/${ticker}`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        setTickerData(data);
        console.log('Fetched data:', data);
      } catch (err) {
        console.error('Error fetching ticker data:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchTickerData();
  }, [ticker, contextTickerData, isSearchPage]);

  // Handle search - OPTIMIZED (only ticker and name, no descriptions)
  const handleSearch = (query) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    
    const lowerQuery = query.toLowerCase();
    const results = [];
    const seenTickers = new Set();

    // Search in companies.csv first - ONLY ticker and name
    if (companiesLoaded && companiesData.length > 0) {
      companiesData.forEach(company => {
        if (seenTickers.has(company.Ticker)) return;
        
        const matchesTicker = company.Ticker && company.Ticker.toLowerCase().includes(lowerQuery);
        const matchesName = company.Name && company.Name.toLowerCase().includes(lowerQuery);
        
        if (matchesTicker || matchesName) {
          const contextMatch = data.find(row => row.Ticker === company.Ticker);
          
          results.push({
            Ticker: company.Ticker,
            Company_Name: company.Name,
            Current_Price: contextMatch?.Current_Price,
            Change_1D: contextMatch?.Change_1D,
            fromCompaniesCSV: true
          });
          seenTickers.add(company.Ticker);
        }
      });
    }

    // Also search in context data for any additional matches
    data.forEach(row => {
      if (seenTickers.has(row.Ticker)) return;
      
      const matchesTicker = row.Ticker.toLowerCase().includes(lowerQuery);
      const matchesName = row.Company_Name && row.Company_Name.toLowerCase().includes(lowerQuery);
      
      if (matchesTicker || matchesName) {
        results.push(row);
        seenTickers.add(row.Ticker);
      }
    });

    // Sort by relevance
    results.sort((a, b) => {
      const aTickerLower = a.Ticker.toLowerCase();
      const bTickerLower = b.Ticker.toLowerCase();
      
      if (aTickerLower === lowerQuery) return -1;
      if (bTickerLower === lowerQuery) return 1;
      
      if (aTickerLower.startsWith(lowerQuery) && !bTickerLower.startsWith(lowerQuery)) return -1;
      if (bTickerLower.startsWith(lowerQuery) && !aTickerLower.startsWith(lowerQuery)) return 1;
      
      return aTickerLower.localeCompare(bTickerLower);
    });

    setSearchResults(results.slice(0, 15));
    setIsSearching(false);
  };

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/ticker/${searchQuery.toUpperCase()}`);
    }
  };

  const getScoreColor = (score) => {
    if (!score || typeof score !== 'number') return 'text-slate-400';
    if (score >= 0.7) return 'text-emerald-400';
    if (score >= 0.5) return 'text-cyan-400';
    if (score >= 0.3) return 'text-amber-400';
    return 'text-red-400';
  };

  // ========== SEARCH PAGE ==========
  if (isSearchPage) {
    // Show loading state while companies CSV is loading
    if (!companiesLoaded) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-cyan-400 mx-auto mb-4"></div>
            <p className="text-slate-400 text-lg">Loading company database...</p>
            <p className="text-slate-500 text-sm mt-2">This may take a moment</p>
          </div>
        </div>
      );
    }

    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
        <div className="max-w-4xl mx-auto">
          <Link
            to={from === "daily-report" ? "/daily-report" : "/"}
            className="mb-6 flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors"
          >
            <ChevronRight className="w-5 h-5 rotate-180" />
            {from === "daily-report" ? "Back to Daily Report" : "Back to Home"}
          </Link>

          {/* Search Header */}
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700 mb-6 shadow-xl">
            <h1 className="text-4xl font-bold text-cyan-400 mb-2">Ticker Search</h1>
            <p className="text-slate-400">Search for any stock by ticker symbol or company name</p>
          </div>

          {/* Search Bar */}
          <form onSubmit={handleSearchSubmit} className="mb-6">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-400 w-6 h-6" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  handleSearch(e.target.value);
                }}
                placeholder="Enter ticker symbol (e.g., AAPL) or company name..."
                className="w-full pl-14 pr-12 py-4 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 transition-all text-lg"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => {
                    setSearchQuery('');
                    setSearchResults([]);
                  }}
                  className="absolute right-4 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-slate-200 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              )}
            </div>
            <p className="mt-2 text-sm text-slate-500">
              Press Enter to search any ticker, or select from suggestions below
            </p>
          </form>

          {/* Search Results */}
          {searchQuery && (
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl border border-slate-700 shadow-xl overflow-hidden">
              <div className="p-4 border-b border-slate-700">
                <h3 className="font-semibold text-cyan-400">
                  {searchResults.length > 0 
                    ? `Found ${searchResults.length} results` 
                    : 'No results found'}
                </h3>
              </div>
              
              {searchResults.length > 0 ? (
                <div className="divide-y divide-slate-700 max-h-96 overflow-y-auto">
                  {searchResults.map((result) => (
                    <Link
                      key={result.Ticker}
                      to={`/ticker/${result.Ticker}?from=home`}
                      className="block p-4 hover:bg-slate-800/50 transition-colors"
                    >
                      <div className="flex items-center justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <div className="font-semibold text-cyan-400 text-lg">{result.Ticker}</div>
                          <div className="text-slate-300 text-sm">{result.Company_Name || 'N/A'}</div>
                        </div>
                        <div className="text-right flex-shrink-0">
                          {typeof result.Current_Price === 'number' ? (
                            <>
                              <div className="text-slate-200 font-semibold">
                                ${result.Current_Price.toFixed(2)}
                              </div>
                              {typeof result.Change_1D === 'number' && (
                                <div className={`text-sm font-semibold ${result.Change_1D >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                  {result.Change_1D > 0 ? '+' : ''}{result.Change_1D.toFixed(2)}%
                                </div>
                              )}
                            </>
                          ) : (
                            <div className="text-slate-500 text-sm">No price data</div>
                          )}
                        </div>
                      </div>
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="p-8 text-center">
                  <div className="text-slate-500 mb-4">
                    <Search className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p className="text-lg">No tickers found</p>
                    <p className="text-sm mt-2">Press Enter to search for "{searchQuery}" anyway</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Instructions */}
          {!searchQuery && (
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700 shadow-xl">
              <h3 className="font-semibold text-cyan-400 mb-4 text-xl">How to Search</h3>
              <div className="space-y-3 text-slate-300">
                <div className="flex items-start gap-3">
                  <div className="bg-cyan-500/10 rounded-lg p-2 mt-1">
                    <Search className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <div className="font-semibold mb-1">By Ticker Symbol</div>
                    <div className="text-sm text-slate-400">
                      Enter any ticker symbol (e.g., AAPL, TSLA, GOOGL) and press Enter
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-cyan-500/10 rounded-lg p-2 mt-1">
                    <Search className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <div className="font-semibold mb-1">By Company Name</div>
                    <div className="text-sm text-slate-400">
                      Start typing a company name to see matching tickers
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-cyan-500/10 rounded-lg p-2 mt-1">
                    <TrendingUp className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <div className="font-semibold mb-1">Live Data</div>
                    <div className="text-sm text-slate-400">
                      Get real-time stock data and multi-timeframe trading analysis
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // ========== LOADING/ERROR STATES ==========
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-cyan-400 mx-auto mb-4"></div>
          <p className="text-slate-400 text-lg">Loading {ticker} data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-400 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-slate-200 mb-2">Error Loading Data</h2>
          <p className="text-slate-400 mb-4">{error}</p>
          <div className="flex gap-4 justify-center">
            <Link
              to="/ticker/__srch__"
              className="inline-flex items-center gap-2 px-6 py-3 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 transition-colors"
            >
              <Search className="w-5 h-5" />
              Search Another Ticker
            </Link>
            <Link
              to={from === "daily-report" ? "/daily-report" : "/"}
              className="inline-flex items-center gap-2 px-6 py-3 bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition-colors"
            >
              <ChevronRight className="w-5 h-5 rotate-180" />
              {from === "daily-report" ? "Back to Daily Report" : "Back to Home"}
            </Link>
          </div>
        </div>
      </div>
    );
  }

  if (!tickerData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="text-slate-600 text-6xl mb-4">📊</div>
          <h2 className="text-2xl font-bold text-slate-200 mb-2">No Data Available</h2>
          <p className="text-slate-400 mb-4">Could not find data for {ticker}</p>
          <div className="flex gap-4 justify-center">
            <Link
              to="/ticker/__srch__"
              className="inline-flex items-center gap-2 px-6 py-3 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 transition-colors"
            >
              <Search className="w-5 h-5" />
              Search Another Ticker
            </Link>
            <Link
              to= {from === "daily-report" ? "/daily-report" : "/"}
              className="inline-flex items-center gap-2 px-6 py-3 bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition-colors"
            >
              <ChevronRight className="w-5 h-5 rotate-180" />
              {from === "daily-report" ? "Back to Daily Report" : "Back to Home"}
            </Link>
          </div>
        </div>
      </div>
    );
  }

  // ========== TICKER DETAIL PAGE ==========
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <Link
            to={from === "daily-report" ? "/daily-report" : "/"}
            className="flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors"
          >
            <ChevronRight className="w-5 h-5 rotate-180" />
            {from === "daily-report" ? "Back to Daily Report" : "Back to Home"}
          </Link>
          <Link
            to="/ticker/__srch__"
            className="flex items-center gap-2 px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-300 hover:text-cyan-400 hover:border-cyan-400 transition-colors"
          >
            <Search className="w-4 h-4" />
            Search Ticker
          </Link>
        </div>

        {/* Header */}
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700 mb-6 shadow-xl">
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-4xl font-bold text-cyan-400 mb-2">{tickerData.Ticker}</h1>
              <p className="text-xl text-slate-300 mb-4">{tickerData.Company_Name || 'N/A'}</p>
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
                  {typeof tickerData.Profit_Margin === 'number' ? `${(tickerData.Profit_Margin * 100).toFixed(2)}%` : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                <span className="text-slate-300 font-medium">ROE</span>
                <span className="font-semibold text-slate-200">
                  {typeof tickerData.ROE === 'number' ? `${(tickerData.ROE * 100).toFixed(2)}%` : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};