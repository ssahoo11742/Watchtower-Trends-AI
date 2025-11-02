// src/pages/CustomJobsPage.jsx
import { Link } from 'react-router-dom';
import { Home, Settings } from 'lucide-react';

export const CustomJobsPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      <div className="max-w-4xl mx-auto">
        <Link
          to="/"
          className="mb-6 flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors"
        >
          <Home className="w-5 h-5" />
          Back to Home
        </Link>
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
}