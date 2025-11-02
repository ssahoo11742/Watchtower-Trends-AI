import React, { useState, createContext, useContext } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useParams, useNavigate } from 'react-router-dom';
import { Upload, ChevronDown, ChevronRight, TrendingUp, TrendingDown, Home, FileText, Settings, Search, BarChart3, Activity, Zap, Globe, Shield } from 'lucide-react';
import Papa from 'papaparse';
import { useData } from '../../contexts/DataContext';

// ==================== HOME PAGE ====================
export const HomePage = () => {
  const { data, handleFileUpload } = useData();

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