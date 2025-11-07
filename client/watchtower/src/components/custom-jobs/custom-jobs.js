import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Papa from 'papaparse';
import { Home, Settings, Play, Calendar, Search, Filter, ChevronDown, ChevronRight, Clock, CheckCircle, XCircle, Loader, Plus, X as XIcon, HelpCircle, FileText, Download, Eye } from 'lucide-react';
import { createClient } from '@supabase/supabase-js';
import { useData } from '../../contexts/DataContext';

// Initialize Supabase client
const supabaseUrl = process.env.REACT_APP_SUPABASE_URL || '';
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY || '';
const supabase = createClient(supabaseUrl, supabaseAnonKey);

const PRESET_TOPICS = [
  { value: "healthcare", label: "Healthcare & Biotech", query: "Pharmacy OR healthcare OR biotech OR biopharmaceutical" },
  { value: "financial", label: "Financial Services", query: "financial OR banking OR insurance OR fintech" },
  { value: "real_estate", label: "Real Estate & Property", query: "real estate OR property OR housing" },
  { value: "retail", label: "Retail & E-commerce", query: "retail OR e-commerce OR consumer goods" },
  { value: "defense", label: "Defense & Military", query: "defense OR military OR weapons" },
  { value: "space", label: "Space & Aerospace", query: "space OR aerospace OR satellite" },
  { value: "technology", label: "Technology & Telecom", query: "technology OR innovation OR 5g OR telecom" },
  { value: "ai", label: "AI & Robotics", query: "ai OR artificial intelligence OR robotics" },
  { value: "energy", label: "Energy & Climate", query: "energy OR renewable OR climate" },
  { value: "cybersecurity", label: "Cybersecurity & Drones", query: "cybersecurity OR drone" },
  { value: "quantum", label: "Quantum & Semiconductors", query: "Quantum OR Post-Quantum OR Quantum Cryptography OR semiconductor" },
];

const Tooltip = ({ children, text }) => {
  const [show, setShow] = useState(false);
  
  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="cursor-help"
      >
        {children}
      </div>
      {show && (
        <div className="absolute z-50 w-72 p-3 bg-slate-700 border border-slate-600 rounded-lg shadow-xl text-xs text-slate-300 left-0 top-full mt-2">
          {text}
        </div>
      )}
    </div>
  );
};

export const CustomJobsPage = () => {
  const navigate = useNavigate();
  const { handleFileUpload } = useData();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [userJobs, setUserJobs] = useState([]);
  const [loadingJobs, setLoadingJobs] = useState(false);
  const [useCustomQueries, setUseCustomQueries] = useState(false);
  const [loadingReport, setLoadingReport] = useState(null);

  // Form state
  const [jobName, setJobName] = useState('');
  const [fromDate, setFromDate] = useState('');
  const [toDate, setToDate] = useState('');
  const [selectedTopics, setSelectedTopics] = useState([]);
  const [customQueries, setCustomQueries] = useState([{ id: 1, query: '' }]);
  const [minTopicSize, setMinTopicSize] = useState(8);
  const [topNCompanies, setTopNCompanies] = useState(50);
  const [minArticles, setMinArticles] = useState(10);
  const [maxArticles, setMaxArticles] = useState(500);

  useEffect(() => {
    checkUser();
    // Set default dates (last 30 days)
    const today = new Date();
    const thirtyDaysAgo = new Date(today);
    thirtyDaysAgo.setDate(today.getDate() - 30);
    
    setToDate(today.toISOString().split('T')[0]);
    setFromDate(thirtyDaysAgo.toISOString().split('T')[0]);
  }, []);

  useEffect(() => {
    if (user) {
      // Poll for job updates every 10 seconds
      const interval = setInterval(() => {
        fetchUserJobs(user.id);
      }, 10000);
      
      return () => clearInterval(interval);
    }
  }, [user]);

  const checkUser = async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      
      if (!session) {
        navigate('/auth');
        return;
      }

      setUser(session.user);
      await fetchUserJobs(session.user.id);
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchUserJobs = async (userId) => {
    try {
      setLoadingJobs(true);
      const { data, error } = await supabase
        .from('custom_jobs')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false });

      if (error) throw error;
      setUserJobs(data || []);
    } catch (err) {
      console.error('Error fetching jobs:', err);
    } finally {
      setLoadingJobs(false);
    }
  };

  const toggleTopic = (topic) => {
    setSelectedTopics(prev => 
      prev.includes(topic)
        ? prev.filter(t => t !== topic)
        : [...prev, topic]
    );
  };

  const addCustomQuery = () => {
    setCustomQueries([...customQueries, { id: Date.now(), query: '' }]);
  };

  const removeCustomQuery = (id) => {
    if (customQueries.length > 1) {
      setCustomQueries(customQueries.filter(q => q.id !== id));
    }
  };

  const updateCustomQuery = (id, value) => {
    setCustomQueries(customQueries.map(q => 
      q.id === id ? { ...q, query: value } : q
    ));
  };

  const handleSubmit = async () => {
    if (!jobName.trim()) {
      alert('Please enter a job name');
      return;
    }

    let queries = [];
    if (useCustomQueries) {
      const validQueries = customQueries.filter(q => q.query.trim());
      if (validQueries.length === 0) {
        alert('Please enter at least one search query');
        return;
      }
      queries = validQueries.map(q => q.query.trim());
    } else {
      if (selectedTopics.length === 0) {
        alert('Please select at least one topic');
        return;
      }
      queries = selectedTopics.map(topic => 
        PRESET_TOPICS.find(t => t.value === topic)?.query
      );
    }

    try {
      const jobConfig = {
        job_name: jobName,
        from_date: fromDate,
        to_date: toDate,
        query_type: useCustomQueries ? 'custom' : 'preset',
        topics: useCustomQueries ? [] : selectedTopics,
        custom_queries: useCustomQueries ? queries : [],
        queries: queries,
        min_topic_size: minTopicSize,
        top_n_companies: topNCompanies,
        min_articles: minArticles,
        max_articles: maxArticles
      };

      // Save to Supabase with pending status
      const { data: savedJob, error: saveError } = await supabase
        .from('custom_jobs')
        .insert([{
          user_id: user.id,
          job_name: jobName,
          config: jobConfig,
          status: 'pending',
          created_at: new Date().toISOString()
        }])
        .select()
        .single();

      if (saveError) throw saveError;

      // Call API endpoint (fire and forget - don't wait for response)
      const apiUrl = process.env.REACT_APP_API_URL || 'https://watchtower-trends-ai.onrender.com';
      fetch(`${apiUrl}/custom_job`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_id: savedJob.id,
          user_id: user.id,
          ...jobConfig
        })
      }).catch(err => {
        console.error('API call failed (background):', err);
      });

      // Refresh jobs list
      await fetchUserJobs(user.id);

      // Reset form
      setJobName('');
      setSelectedTopics([]);
      setCustomQueries([{ id: 1, query: '' }]);
      
      alert('Job submitted successfully! It will appear in "Your Jobs" with status updates.');
    } catch (err) {
      console.error('Error submitting job:', err);
      alert('Failed to submit job: ' + err.message);
    }
  };

  const loadJobReport = async (job) => {
    if (!job.result_file_path) {
      alert('No result file available yet');
      return;
    }

    try {
      setLoadingReport(job.id);
      
      // Get the file from Supabase storage
      const { data: urlData } = supabase.storage
        .from('daily-reports')
        .getPublicUrl(job.result_file_path);

      if (!urlData?.publicUrl) {
        throw new Error('Failed to get public URL');
      }

      const response = await fetch(urlData.publicUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch file: ${response.status}`);
      }
      
      const text = await response.text();
      
      Papa.parse(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          const blob = new Blob([text], { type: 'text/csv' });
          const file = new File([blob], job.result_file_path.split('/').pop(), { type: 'text/csv' });
          const event = { target: { files: [file] } };
          handleFileUpload(event);
          
          setTimeout(() => {
            setLoadingReport(null);
            navigate('/daily-report');
          }, 500);
        },
        error: (error) => {
          console.error('Error parsing CSV:', error);
          alert('Failed to parse CSV: ' + error.message);
          setLoadingReport(null);
        }
      });
    } catch (err) {
      console.error('Error loading report:', err);
      alert('Failed to load report: ' + err.message);
      setLoadingReport(null);
    }
  };

  const downloadJobReport = async (job) => {
    if (!job.result_file_path) {
      alert('No result file available yet');
      return;
    }

    try {
      const fileName = job.result_file_path.split('/').pop();
      const { data: urlData } = supabase.storage
        .from('daily-reports')
        .getPublicUrl(job.result_file_path);

      if (urlData?.publicUrl) {
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

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-400" />;
      case 'running':
        return <Loader className="w-5 h-5 text-blue-400 animate-spin" />;
      default:
        return <Clock className="w-5 h-5 text-yellow-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500/20 text-green-400 border-green-500';
      case 'failed':
        return 'bg-red-500/20 text-red-400 border-red-500';
      case 'running':
        return 'bg-blue-500/20 text-blue-400 border-blue-500';
      default:
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="inline-block w-12 h-12 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      <div className="max-w-6xl mx-auto">
        <Link
          to="/"
          className="mb-6 flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors"
        >
          <Home className="w-5 h-5" />
          Back to Home
        </Link>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Create Job */}
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700">
            <div className="flex items-center gap-3 mb-6">
              <Settings className="w-8 h-8 text-purple-400" />
              <h1 className="text-3xl font-bold text-purple-400">Create Custom Job</h1>
            </div>

            <div className="space-y-6">
              {/* Job Name */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Job Name *
                </label>
                <input
                  type="text"
                  value={jobName}
                  onChange={(e) => setJobName(e.target.value)}
                  placeholder="e.g., Q4 Healthcare Analysis"
                  className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>

              {/* Date Range */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-2">
                    From Date *
                    <Tooltip text="The start date for fetching news articles. The system will analyze articles published from this date onwards.">
                      <HelpCircle className="w-4 h-4 text-slate-500" />
                    </Tooltip>
                  </label>
                  <input
                    type="date"
                    value={fromDate}
                    onChange={(e) => setFromDate(e.target.value)}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-2">
                    To Date *
                    <Tooltip text="The end date for fetching news articles. Analysis will include all articles up to and including this date.">
                      <HelpCircle className="w-4 h-4 text-slate-500" />
                    </Tooltip>
                  </label>
                  <input
                    type="date"
                    value={toDate}
                    onChange={(e) => setToDate(e.target.value)}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
              </div>

              {/* Query Type Toggle */}
              <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-slate-300">
                    Search Query Type
                  </label>
                  <button
                    onClick={() => setUseCustomQueries(!useCustomQueries)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      useCustomQueries
                        ? 'bg-purple-600 text-white'
                        : 'bg-slate-600 text-slate-300'
                    }`}
                  >
                    {useCustomQueries ? 'Custom Queries' : 'Preset Topics'}
                  </button>
                </div>
                <p className="text-xs text-slate-400">
                  {useCustomQueries 
                    ? 'Write your own search queries using AND, OR, and parentheses for complex searches'
                    : 'Select from predefined topic categories'}
                </p>
              </div>

              {/* Conditional Rendering: Preset Topics or Custom Queries */}
              {!useCustomQueries ? (
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-3">
                    Select Topics * ({selectedTopics.length} selected)
                    <Tooltip text="Choose one or more topic categories to analyze. Each topic represents a broad industry or sector with predefined search terms.">
                      <HelpCircle className="w-4 h-4 text-slate-500" />
                    </Tooltip>
                  </label>
                  <div className="space-y-2 max-h-64 overflow-y-auto bg-slate-700/30 rounded-lg p-3">
                    {PRESET_TOPICS.map(topic => (
                      <button
                        key={topic.value}
                        onClick={() => toggleTopic(topic.value)}
                        className={`w-full text-left px-4 py-2 rounded-lg transition-all ${
                          selectedTopics.includes(topic.value)
                            ? 'bg-purple-600 text-white'
                            : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span>{topic.label}</span>
                          {selectedTopics.includes(topic.value) && (
                            <CheckCircle className="w-4 h-4" />
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <label className="flex items-center gap-2 text-sm font-medium text-slate-300">
                      Custom Search Queries *
                      <Tooltip text="Write custom search queries using boolean operators. Use 'OR' to match any term, 'AND' to match all terms, and parentheses to group conditions.">
                        <HelpCircle className="w-4 h-4 text-slate-500" />
                      </Tooltip>
                    </label>
                    <button
                      onClick={addCustomQuery}
                      className="flex items-center gap-1 text-xs bg-purple-600 hover:bg-purple-500 text-white px-3 py-1 rounded-lg transition-all"
                    >
                      <Plus className="w-3 h-3" />
                      Add Query
                    </button>
                  </div>
                  
                  <div className="space-y-3 max-h-64 overflow-y-auto bg-slate-700/30 rounded-lg p-3">
                    {customQueries.map((q, index) => (
                      <div key={q.id} className="flex items-start gap-2">
                        <div className="flex-1">
                          <input
                            type="text"
                            value={q.query}
                            onChange={(e) => updateCustomQuery(q.id, e.target.value)}
                            placeholder="e.g., (Tesla OR SpaceX) AND (Musk OR innovation)"
                            className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                          />
                          <p className="text-xs text-slate-500 mt-1">
                            Use OR, AND, and parentheses
                          </p>
                        </div>
                        {customQueries.length > 1 && (
                          <button
                            onClick={() => removeCustomQuery(q.id)}
                            className="mt-1 p-2 text-red-400 hover:text-red-300 hover:bg-red-900/20 rounded-lg transition-all"
                          >
                            <XIcon className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Advanced Settings */}
              <div>
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors"
                >
                  {showAdvanced ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
                  Advanced Settings
                </button>

                {showAdvanced && (
                  <div className="mt-4 space-y-4 pl-4 border-l-2 border-slate-700">
                    <div>
                      <label className="flex items-center gap-2 text-sm font-medium text-slate-400 mb-2">
                        Minimum Topic Size: {minTopicSize}
                        <Tooltip text="The minimum number of articles required to form a topic cluster.">
                          <HelpCircle className="w-3 h-3 text-slate-500" />
                        </Tooltip>
                      </label>
                      <input
                        type="range"
                        min="5"
                        max="20"
                        value={minTopicSize}
                        onChange={(e) => setMinTopicSize(parseInt(e.target.value))}
                        className="w-full"
                      />
                    </div>

                    <div>
                      <label className="flex items-center gap-2 text-sm font-medium text-slate-400 mb-2">
                        Top N Companies: {topNCompanies}
                        <Tooltip text="Maximum number of companies to include in results.">
                          <HelpCircle className="w-3 h-3 text-slate-500" />
                        </Tooltip>
                      </label>
                      <input
                        type="range"
                        min="10"
                        max="100"
                        step="10"
                        value={topNCompanies}
                        onChange={(e) => setTopNCompanies(parseInt(e.target.value))}
                        className="w-full"
                      />
                    </div>

                    <div>
                      <label className="flex items-center gap-2 text-sm font-medium text-slate-400 mb-2">
                        Min Articles: {minArticles}
                        <Tooltip text="Minimum number of articles required before analysis.">
                          <HelpCircle className="w-3 h-3 text-slate-500" />
                        </Tooltip>
                      </label>
                      <input
                        type="range"
                        min="5"
                        max="50"
                        step="5"
                        value={minArticles}
                        onChange={(e) => setMinArticles(parseInt(e.target.value))}
                        className="w-full"
                      />
                    </div>

                    <div>
                      <label className="flex items-center gap-2 text-sm font-medium text-slate-400 mb-2">
                        Max Articles: {maxArticles}
                        <Tooltip text="Maximum number of articles to process.">
                          <HelpCircle className="w-3 h-3 text-slate-500" />
                        </Tooltip>
                      </label>
                      <input
                        type="range"
                        min="100"
                        max="1000"
                        step="100"
                        value={maxArticles}
                        onChange={(e) => setMaxArticles(parseInt(e.target.value))}
                        className="w-full"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Submit Button */}
              <button
                onClick={handleSubmit}
                className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-semibold py-4 rounded-lg transition-all shadow-lg hover:shadow-purple-500/50 flex items-center justify-center gap-2"
              >
                <Play className="w-5 h-5" />
                Submit Job
              </button>
            </div>
          </div>

          {/* Right Column - Job History */}
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 border border-slate-700">
            <h2 className="text-2xl font-bold text-cyan-400 mb-6">Your Jobs</h2>

            {loadingJobs ? (
              <div className="text-center py-12">
                <div className="inline-block w-8 h-8 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
              </div>
            ) : userJobs.length === 0 ? (
              <div className="text-center py-12 text-slate-400">
                <Filter className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No jobs yet. Create your first custom analysis!</p>
              </div>
            ) : (
              <div className="space-y-4 max-h-[600px] overflow-y-auto">
                {userJobs.map(job => (
                  <div
                    key={job.id}
                    className="bg-slate-700/30 rounded-lg p-4 border border-slate-600 hover:border-cyan-500 transition-all"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <h3 className="text-lg font-semibold text-slate-200">{job.job_name}</h3>
                      <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(job.status)}`}>
                        {getStatusIcon(job.status)}
                        {job.status}
                      </div>
                    </div>

                    <div className="space-y-2 text-sm">
                      <div className="flex items-center gap-2 text-slate-400">
                        <Calendar className="w-4 h-4" />
                        {job.config.from_date} to {job.config.to_date}
                      </div>
                      <div className="flex items-center gap-2 text-slate-400">
                        <Search className="w-4 h-4" />
                        {job.config.query_type === 'custom' 
                          ? `${job.config.custom_queries?.length || 0} custom queries`
                          : `${job.config.topics?.length || 0} preset topics`
                        }
                      </div>
                      <div className="text-slate-500 text-xs">
                        Created: {new Date(job.created_at).toLocaleString()}
                      </div>

                      {job.status === 'completed' && job.result_file_path && (
                        <div className="mt-3 pt-3 border-t border-slate-600">
                          <div className="flex gap-2">
                            <button
                              onClick={() => loadJobReport(job)}
                              disabled={loadingReport === job.id}
                              className="flex-1 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white px-3 py-2 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                            >
                              {loadingReport === job.id ? (
                                <>
                                  <Loader className="w-4 h-4 animate-spin" />
                                  Loading...
                                </>
                              ) : (
                                <>
                                  <Eye className="w-4 h-4" />
                                  View Report
                                </>
                              )}
                            </button>
                            <button
                              onClick={() => downloadJobReport(job)}
                              className="bg-slate-600 hover:bg-slate-500 text-white px-3 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2"
                            >
                              <Download className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      )}

                      {job.status === 'failed' && job.error_message && (
                        <div className="mt-2 text-xs text-red-400 bg-red-900/20 rounded p-2">
                          Error: {job.error_message}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};