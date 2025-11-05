import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useParams, useNavigate } from 'react-router-dom';
import { Upload, ChevronDown, ChevronRight, TrendingUp, TrendingDown, Home, FileText, Settings, Search, BarChart3, Activity, Zap, Globe, Shield, Download, Calendar } from 'lucide-react';
import Papa from 'papaparse';
import { useData } from '../../contexts/DataContext';
import { createClient } from '@supabase/supabase-js';
import { useNavigate as useNav } from 'react-router-dom';

// Initialize Supabase client
const supabaseUrl = process.env.REACT_APP_SUPABASE_URL || '';
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY || '';
const supabase = createClient(supabaseUrl, supabaseAnonKey);

// ==================== HOME PAGE ====================
export const HomePage = () => {
  const { data, handleFileUpload } = useData();
  const navigate = useNav();
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [loadingReport, setLoadingReport] = useState(false);

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      setLoading(true);
      setError(null);

      // List all files in the daily-reports/report folder
      const { data: files, error: listError } = await supabase.storage
        .from('daily-reports')
        .list('reports', {
          limit: 100,
          offset: 0,
          sortBy: { column: 'created_at', order: 'desc' }
        });

      if (listError) throw listError;

      // Parse filenames to extract date, depth, and timestamp
      const parsedReports = files
        .filter(file => file.name.endsWith('.csv'))
        .map(file => {
          // Expected format: {filename}_depth-{depth}_{MM-DD-YYYY_HH}.csv
          const match = file.name.match(/(.+)_depth-(\d+)_(\d{2}-\d{2}-\d{4}_\d{2})\.csv$/);
          
          if (match) {
            const [, baseName, depth, timestamp] = match;
            
            // Parse timestamp: MM-DD-YYYY_HH
            const [datePart, hourPart] = timestamp.split('_');
            const [month, day, year] = datePart.split('-');
            const hour = hourPart;
            
            // Create date object (note: month is 0-indexed in JS Date)
            const date = new Date(parseInt(year), parseInt(month) - 1, parseInt(day), parseInt(hour), 0, 0);
            
            return {
              name: file.name,
              baseName,
              depth: parseInt(depth),
              timestamp,
              date,
              dateString: date.toLocaleDateString('en-US', { 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
              }),
              size: file.metadata?.size || 0
            };
          }
          return null;
        })
        .filter(Boolean)
        .sort((a, b) => b.date - a.date); // Sort by date descending

      setReports(parsedReports);
    } catch (err) {
      console.error('Error fetching reports:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async (fileName) => {
    try {
      // Get public URL instead of downloading directly
      const { data: urlData } = supabase.storage
        .from('daily-reports')
        .getPublicUrl(`reports/${fileName}`);

      if (urlData?.publicUrl) {
        // Use fetch to download the file
        const response = await fetch(urlData.publicUrl);
        if (!response.ok) throw new Error('Failed to fetch file');
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } else {
        throw new Error('Failed to get public URL');
      }
    } catch (err) {
      console.error('Error downloading report:', err);
      alert('Failed to download report: ' + err.message);
    }
  };

  const loadReportData = async (fileName) => {
    try {
      setLoadingReport(true);
      
      // Get public URL instead of downloading directly
      const { data: urlData } = supabase.storage
        .from('daily-reports')
        .getPublicUrl(`reports/${fileName}`);

      if (!urlData?.publicUrl) {
        throw new Error('Failed to get public URL');
      }

      // Fetch the file content
      const response = await fetch(urlData.publicUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch file: ${response.status} ${response.statusText}`);
      }
      
      const text = await response.text();
      
      // Parse CSV
      Papa.parse(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          // Create a fake event object to pass to handleFileUpload
          const blob = new Blob([text], { type: 'text/csv' });
          const file = new File([blob], fileName, { type: 'text/csv' });
          const event = { target: { files: [file] } };
          handleFileUpload(event);
          
          // Navigate to daily report page after data is loaded
          setTimeout(() => {
            setLoadingReport(false);
            navigate('/daily-report');
          }, 500);
        },
        error: (error) => {
          console.error('Error parsing CSV:', error);
          alert('Failed to parse CSV: ' + error.message);
          setLoadingReport(false);
        }
      });
    } catch (err) {
      console.error('Error loading report:', err);
      alert('Failed to load report: ' + err.message);
      setLoadingReport(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
       {/* Hero Section */}
        <div className="text-center mb-16 pt-12">
          <div className="flex flex-col items-center gap-6 mb-6">
            {/* Logo */}
            <img 
              src={`${process.env.PUBLIC_URL}/logo_nobg_notext.png`}
              alt="Watchtower Trends AI Logo" 
              className="h-64 w-auto object-contain brightness-110 contrast-125"
              style={{
                filter: 'brightness(1.1) contrast(1.25) saturate(1.4) hue-rotate(-5deg)'
              }}
              onError={(e) => {
                // Fallback to inline SVG if image fails to load
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'block';
              }}
            />
            {/* Fallback SVG Logo */}
            <svg 
              className="h-48 w-auto hidden" 
              viewBox="0 0 200 240" 
              fill="none" 
              xmlns="http://www.w3.org/2000/svg"
              preserveAspectRatio="xMidYMid meet"
            >
              {/* Signal waves */}
              <path d="M60 40 Q80 30 100 40" stroke="#22D3EE" strokeWidth="8" strokeLinecap="round" fill="none"/>
              <path d="M70 50 Q85 43 100 50" stroke="#22D3EE" strokeWidth="8" strokeLinecap="round" fill="none"/>
              <path d="M140 40 Q120 30 100 40" stroke="#22D3EE" strokeWidth="8" strokeLinecap="round" fill="none"/>
              <path d="M130 50 Q115 43 100 50" stroke="#22D3EE" strokeWidth="8" strokeLinecap="round" fill="none"/>
              
              {/* Eye */}
              <ellipse cx="100" cy="85" rx="35" ry="25" fill="none" stroke="#22D3EE" strokeWidth="6"/>
              <circle cx="100" cy="85" r="12" fill="#22D3EE"/>
              
              {/* Tower structure */}
              <rect x="90" y="100" width="20" height="15" fill="#22D3EE"/>
              <path d="M75 115 L100 140 L125 115 L115 115 L100 125 L85 115 Z" fill="#22D3EE"/>
              <path d="M80 140 L100 160 L120 140 L112 140 L100 150 L88 140 Z" fill="#22D3EE"/>
              <rect x="95" y="160" width="10" height="30" fill="#22D3EE"/>
            </svg>
            
            <h1 className="text-7xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent">
              WATCHTOWER
            </h1>
            <p className="text-xl font-semibold text-cyan-400 -mt-4">TRENDS AI</p>
          </div>
          <p className="text-2xl text-slate-300 mb-4">AI-Powered Stock Analysis Platform</p>
          <p className="text-lg text-slate-400 max-w-3xl mx-auto">
            Leveraging machine learning and real-time news analysis to identify trending industries and investment opportunities
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <Link
            to="/daily-report"
            className="group bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700 hover:border-cyan-500 transition-all hover:shadow-2xl hover:shadow-cyan-500/20 text-left block"
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
          </Link>

          <Link
            to="/custom-jobs"
            className="group bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700 hover:border-purple-500 transition-all hover:shadow-2xl hover:shadow-purple-500/20 text-left block"
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
          </Link>

          
          <Link to="/ticker/__srch__?from=home" className="group bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700 hover:border-blue-500 transition-all hover:shadow-2xl hover:shadow-blue-500/20">
            <div className="bg-gradient-to-br from-blue-500 to-indigo-600 w-16 h-16 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Search className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-2xl font-bold text-blue-400 mb-3">Ticker Search</h3>
            <p className="text-slate-400 mb-4">
              Deep dive into individual stocks with comprehensive charts, fundamentals, and technical analysis
            </p>
            <div className="flex items-center text-blue-500 font-medium">
              Search Tickers
              <ChevronRight className="w-5 h-5 ml-2 group-hover:translate-x-2 transition-transform" />
            </div>
          </Link>
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

        {/* Daily Reports Table */}
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700 mb-12">
          <div className="flex items-center gap-3 mb-6">
            <Calendar className="w-8 h-8 text-cyan-400" />
            <h2 className="text-3xl font-bold text-slate-200">Available Daily Reports</h2>
          </div>

          {loading && (
            <div className="text-center py-12">
              <div className="inline-block w-12 h-12 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
              <p className="text-slate-400 mt-4">Loading reports...</p>
            </div>
          )}

          {error && (
            <div className="bg-red-900/20 border border-red-500 rounded-lg p-4 text-red-400">
              Error loading reports: {error}
            </div>
          )}

          {!loading && !error && reports.length === 0 && (
            <div className="text-center py-12 text-slate-400">
              No reports found in the database.
            </div>
          )}

          {!loading && !error && reports.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left py-4 px-4 text-cyan-400 font-semibold">Report Name</th>
                    <th className="text-left py-4 px-4 text-cyan-400 font-semibold">Date & Time</th>
                    <th className="text-center py-4 px-4 text-cyan-400 font-semibold">Depth</th>
                    <th className="text-right py-4 px-4 text-cyan-400 font-semibold">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {reports.map((report, index) => (
                    <tr 
                      key={report.name} 
                      className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors"
                    >
                      <td className="py-4 px-4 text-slate-300 font-mono text-sm">
                        {report.baseName}
                      </td>
                      <td className="py-4 px-4 text-slate-300">
                        {report.dateString}
                      </td>
                      <td className="py-4 px-4 text-center">
                        <span className="inline-block bg-purple-500/20 text-purple-400 px-3 py-1 rounded-full text-sm font-semibold">
                          {report.depth}
                        </span>
                      </td>
                      <td className="py-4 px-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => loadReportData(report.name)}
                            disabled={loadingReport}
                            className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all shadow-lg hover:shadow-cyan-500/50 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {loadingReport ? (
                              <>
                                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                Loading...
                              </>
                            ) : (
                              <>
                                <FileText className="w-4 h-4" />
                                Load Report
                              </>
                            )}
                          </button>
                          <button
                            onClick={() => downloadReport(report.name)}
                            className="bg-slate-700 hover:bg-slate-600 text-slate-300 px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2"
                          >
                            <Download className="w-4 h-4" />
                            Download
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

{/* Upload Section */}
{data.length === 0 && (
  <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-12 text-center border border-slate-700">
    <Upload className="w-12 h-12 mx-auto mb-4 text-cyan-400" />
    <h2 className="text-2xl font-bold text-slate-200 mb-2">Upload Analysis Data</h2>
    <p className="text-slate-400 mb-6">Upload a CSV file to start exploring the daily report</p>
    <div className="flex items-center justify-center gap-4">
      <label className="cursor-pointer bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium px-6 py-3 rounded-lg transition-all shadow-lg hover:shadow-cyan-500/50">
        <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
        Upload CSV File
      </label>
      <button
        onClick={async () => {
          try {
            const response = await fetch(`${process.env.PUBLIC_URL}/topic_companies_multitimeframe.csv`);
            const csvText = await response.text();
            const blob = new Blob([csvText], { type: 'text/csv' });
            const file = new File([blob], 'topics.csv', { type: 'text/csv' });
            const event = { target: { files: [file] } };
            handleFileUpload(event);
          } catch (error) {
            console.error('Error loading premade data:', error);
            alert('Failed to load premade data. Make sure topics.csv exists in the public folder.');
          }
        }}
        className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-medium px-6 py-3 rounded-lg transition-all shadow-lg hover:shadow-purple-500/50"
      >
        Use Premade Data (TESTING ONLY)
      </button>
    </div>
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
            <Link
              to="/daily-report"
              className="inline-block bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium px-8 py-3 rounded-lg transition-all shadow-lg hover:shadow-cyan-500/50"
            >
              View Daily Report
            </Link>
          </div>
        )}
      </div>
    </div>
  );
};