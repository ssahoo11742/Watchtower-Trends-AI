import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { User, Mail, Calendar, LogOut, ArrowLeft, Edit2, Save, X, Settings, Clock, CheckCircle, XCircle, Loader, Search, Filter, Eye, Download, FileText } from 'lucide-react';
import { createClient } from '@supabase/supabase-js';
import Papa from 'papaparse';
import { useData } from '../../contexts/DataContext';

// Initialize Supabase client
const supabaseUrl = process.env.REACT_APP_SUPABASE_URL || '';
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY || '';
const supabase = createClient(supabaseUrl, supabaseAnonKey);

export const ProfilePage = () => {
  const navigate = useNavigate();
  const { handleFileUpload } = useData();
  const [user, setUser] = useState(null);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [fullName, setFullName] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [customJobs, setCustomJobs] = useState([]);
  const [loadingJobs, setLoadingJobs] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [loadingReport, setLoadingReport] = useState(null);

  useEffect(() => {
    checkUser();
  }, []);

  const checkUser = async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      
      if (!session) {
        navigate('/login');
        return;
      }

      setUser(session.user);
      
      // Fetch profile data
      const { data: profileData, error: profileError } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', session.user.id)
        .single();

      if (profileError && profileError.code !== 'PGRST116') {
        console.error('Error fetching profile:', profileError);
      } else if (profileData) {
        setProfile(profileData);
        setFullName(profileData.full_name || '');
      }

      // Fetch custom jobs
      await fetchCustomJobs(session.user.id);
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchCustomJobs = async (userId) => {
    try {
      setLoadingJobs(true);
      const { data, error } = await supabase
        .from('custom_jobs')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false });

      if (error) throw error;
      setCustomJobs(data || []);
    } catch (err) {
      console.error('Error fetching jobs:', err);
    } finally {
      setLoadingJobs(false);
    }
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    navigate('/');
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      setError('');
      setSuccess('');

      const { error: updateError } = await supabase
        .from('profiles')
        .update({ 
          full_name: fullName,
          updated_at: new Date().toISOString()
        })
        .eq('id', user.id);

      if (updateError) throw updateError;

      setProfile({ ...profile, full_name: fullName });
      setSuccess('Profile updated successfully!');
      setEditing(false);

      setTimeout(() => setSuccess(''), 3000);
    } catch (err) {
      console.error('Error updating profile:', err);
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setFullName(profile?.full_name || '');
    setEditing(false);
    setError('');
  };

  const loadJobReport = async (job) => {
    if (!job.result_file_path) {
      alert('No result file available yet');
      return;
    }

    try {
      setLoadingReport(job.id);
      
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
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-400" />;
      case 'running':
        return <Loader className="w-4 h-4 text-blue-400 animate-spin" />;
      default:
        return <Clock className="w-4 h-4 text-yellow-400" />;
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

  if (!user) {
    return null;
  }

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const formatDateTime = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getJobStats = () => {
    const completed = customJobs.filter(j => j.status === 'completed').length;
    const running = customJobs.filter(j => j.status === 'running').length;
    const failed = customJobs.filter(j => j.status === 'failed').length;
    
    return { completed, running, failed, total: customJobs.length };
  };

  const stats = getJobStats();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Back Button */}
        <button
          onClick={() => navigate('/')}
          className="flex items-center gap-2 text-slate-400 hover:text-cyan-400 mb-8 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          Back to Home
        </button>

        {/* Profile Card */}
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl border border-slate-700 overflow-hidden mb-6">
          {/* Header with gradient */}
          <div className="bg-gradient-to-r from-cyan-600 to-blue-600 h-32"></div>
          
          <div className="px-8 pb-8">
            {/* Profile Picture */}
            <div className="flex items-end justify-between -mt-16 mb-6">
              <div className="w-32 h-32 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center border-4 border-slate-900 shadow-xl">
                <span className="text-5xl text-white font-bold">
                  {user.email?.charAt(0).toUpperCase()}
                </span>
              </div>
              
              {!editing && (
                <button
                  onClick={() => setEditing(true)}
                  className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-slate-300 px-4 py-2 rounded-lg font-medium transition-all"
                >
                  <Edit2 className="w-4 h-4" />
                  Edit Profile
                </button>
              )}
            </div>

            {/* Success/Error Messages */}
            {success && (
              <div className="bg-green-900/20 border border-green-500 rounded-lg p-4 text-green-400 mb-6">
                {success}
              </div>
            )}

            {error && (
              <div className="bg-red-900/20 border border-red-500 rounded-lg p-4 text-red-400 mb-6">
                {error}
              </div>
            )}

            {/* Tabs */}
            <div className="flex gap-4 mb-6 border-b border-slate-700">
              <button
                onClick={() => setActiveTab('overview')}
                className={`pb-3 px-2 font-medium transition-all ${
                  activeTab === 'overview'
                    ? 'text-cyan-400 border-b-2 border-cyan-400'
                    : 'text-slate-400 hover:text-slate-300'
                }`}
              >
                Overview
              </button>
              <button
                onClick={() => setActiveTab('jobs')}
                className={`pb-3 px-2 font-medium transition-all flex items-center gap-2 ${
                  activeTab === 'jobs'
                    ? 'text-cyan-400 border-b-2 border-cyan-400'
                    : 'text-slate-400 hover:text-slate-300'
                }`}
              >
                Custom Jobs
                {customJobs.length > 0 && (
                  <span className="bg-purple-600 text-white text-xs px-2 py-0.5 rounded-full">
                    {customJobs.length}
                  </span>
                )}
              </button>
            </div>

            {/* Tab Content */}
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* Full Name */}
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-slate-400 mb-2">
                    <User className="w-4 h-4" />
                    Full Name
                  </label>
                  {editing ? (
                    <input
                      type="text"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                      placeholder="Enter your full name"
                    />
                  ) : (
                    <p className="text-2xl font-bold text-slate-200">
                      {profile?.full_name || 'Not set'}
                    </p>
                  )}
                </div>

                {/* Email */}
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-slate-400 mb-2">
                    <Mail className="w-4 h-4" />
                    Email Address
                  </label>
                  <p className="text-lg text-slate-200">{user.email}</p>
                  <p className="text-sm text-slate-500 mt-1">Email cannot be changed</p>
                </div>

                {/* Account Created */}
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-slate-400 mb-2">
                    <Calendar className="w-4 h-4" />
                    Account Created
                  </label>
                  <p className="text-lg text-slate-200">
                    {formatDate(user.created_at)}
                  </p>
                </div>

                {/* User ID */}
                <div>
                  <label className="text-sm font-medium text-slate-400 mb-2 block">
                    User ID
                  </label>
                  <p className="text-sm text-slate-400 font-mono bg-slate-700/50 px-4 py-2 rounded-lg break-all">
                    {user.id}
                  </p>
                </div>
              </div>
            )}

            {activeTab === 'jobs' && (
              <div>
                {/* Job Stats Summary */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-slate-700/30 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-1">
                      <Settings className="w-4 h-4 text-slate-400" />
                      <p className="text-xs text-slate-400">Total Jobs</p>
                    </div>
                    <p className="text-2xl font-bold text-cyan-400">{stats.total}</p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-1">
                      <CheckCircle className="w-4 h-4 text-green-400" />
                      <p className="text-xs text-slate-400">Completed</p>
                    </div>
                    <p className="text-2xl font-bold text-green-400">{stats.completed}</p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-1">
                      <Loader className="w-4 h-4 text-blue-400" />
                      <p className="text-xs text-slate-400">Running</p>
                    </div>
                    <p className="text-2xl font-bold text-blue-400">{stats.running}</p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-1">
                      <XCircle className="w-4 h-4 text-red-400" />
                      <p className="text-xs text-slate-400">Failed</p>
                    </div>
                    <p className="text-2xl font-bold text-red-400">{stats.failed}</p>
                  </div>
                </div>

                {/* Jobs List */}
                {loadingJobs ? (
                  <div className="text-center py-12">
                    <div className="inline-block w-8 h-8 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
                    <p className="text-slate-400 mt-4">Loading jobs...</p>
                  </div>
                ) : customJobs.length === 0 ? (
                  <div className="text-center py-12 text-slate-400">
                    <Filter className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p className="mb-4">You haven't created any custom jobs yet.</p>
                    <Link
                      to="/custom-jobs"
                      className="inline-flex items-center gap-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white px-6 py-3 rounded-lg font-medium transition-all shadow-lg hover:shadow-purple-500/50"
                    >
                      <Settings className="w-5 h-5" />
                      Create Your First Job
                    </Link>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {customJobs.map(job => (
                      <div
                        key={job.id}
                        className="bg-slate-700/30 rounded-lg p-5 border border-slate-600 hover:border-cyan-500 transition-all"
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div>
                            <h3 className="text-lg font-semibold text-slate-200 mb-1">
                              {job.job_name}
                            </h3>
                            <p className="text-xs text-slate-500">
                              Created {formatDateTime(job.created_at)}
                            </p>
                          </div>
                          <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(job.status)}`}>
                            {getStatusIcon(job.status)}
                            {job.status}
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-3 text-sm mb-3">
                          <div className="flex items-center gap-2 text-slate-400">
                            <Calendar className="w-4 h-4" />
                            <span className="text-xs">
                              {job.config.from_date} to {job.config.to_date}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-slate-400">
                            <Search className="w-4 h-4" />
                            <span className="text-xs">
                              {job.config.query_type === 'custom' 
                                ? `${job.config.custom_queries?.length || 0} custom queries`
                                : `${job.config.topics?.length || 0} preset topics`
                              }
                            </span>
                          </div>
                        </div>

                        {job.status === 'completed' && job.result_data && (
                          <div className="pt-3 border-t border-slate-600 grid grid-cols-3 gap-2">
                            <div className="text-center bg-cyan-900/20 rounded-lg p-2">
                              <div className="text-lg font-bold text-cyan-400">{job.result_data.companies_analyzed}</div>
                              <div className="text-xs text-slate-400">Companies</div>
                            </div>
                            <div className="text-center bg-purple-900/20 rounded-lg p-2">
                              <div className="text-lg font-bold text-purple-400">{job.result_data.topics_found}</div>
                              <div className="text-xs text-slate-400">Topics</div>
                            </div>
                            <div className="text-center bg-blue-900/20 rounded-lg p-2">
                              <div className="text-lg font-bold text-blue-400">{job.result_data.articles_processed}</div>
                              <div className="text-xs text-slate-400">Articles</div>
                            </div>
                          </div>
                        )}

                        {job.status === 'failed' && job.error_message && (
                          <div className="mt-3 text-xs text-red-400 bg-red-900/20 rounded p-3 border border-red-800">
                            <strong>Error:</strong> {job.error_message}
                          </div>
                        )}

                        {job.status === 'completed' && job.completed_at && (
                          <div className="mt-2 text-xs text-slate-500">
                            Completed {formatDateTime(job.completed_at)}
                          </div>
                        )}

                        {job.status === 'completed' && job.result_file_path && (
                          <div className="mt-3 pt-3 border-t border-slate-600">
                            <div className="flex gap-2">
                              <button
                                onClick={() => loadJobReport(job)}
                                disabled={loadingReport === job.id}
                                className="flex-1 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white px-3 py-2 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
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
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex items-center gap-4 mt-8 pt-8 border-t border-slate-700">
              {editing ? (
                <>
                  <button
                    onClick={handleSave}
                    disabled={saving}
                    className="flex items-center gap-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white px-6 py-3 rounded-lg font-medium transition-all shadow-lg hover:shadow-cyan-500/50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {saving ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        Saving...
                      </>
                    ) : (
                      <>
                        <Save className="w-5 h-5" />
                        Save Changes
                      </>
                    )}
                  </button>
                  <button
                    onClick={handleCancel}
                    disabled={saving}
                    className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-slate-300 px-6 py-3 rounded-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <X className="w-5 h-5" />
                    Cancel
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={handleSignOut}
                    className="flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-medium transition-all shadow-lg hover:shadow-red-500/50"
                  >
                    <LogOut className="w-5 h-5" />
                    Sign Out
                  </button>
                  {activeTab === 'jobs' && customJobs.length > 0 && (
                    <Link
                      to="/custom-jobs"
                      className="flex items-center gap-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white px-6 py-3 rounded-lg font-medium transition-all shadow-lg hover:shadow-purple-500/50"
                    >
                      <Settings className="w-5 h-5" />
                      Create New Job
                    </Link>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};