// src/App.jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { DataProvider } from './contexts/DataContext';
import { HomePage } from './components/home/home';
import { DailyReportPage } from './components/daily-reports/daily-report';
import { CustomJobsPage } from './components/custom-jobs/custom-jobs';
import { TickerDetailPage } from './components/ticker/ticker';
import  AuthPage  from './components/login/login';
import { ProfilePage } from './components/profile/profile';

export default function App() {
  return (
    <DataProvider>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/daily-report" element={<DailyReportPage />} />
          <Route path="/custom-jobs" element={<CustomJobsPage />} />
          <Route path="/ticker/:ticker" element={<TickerDetailPage />} />
          <Route path="/auth" element={<AuthPage />} />
          <Route path="/profile" element={<ProfilePage />} />
        </Routes>
      </Router>
    </DataProvider>
  );
}