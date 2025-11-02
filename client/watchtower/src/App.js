// src/App.jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { DataProvider } from './contexts/DataContext';
import { HomePage } from './components/home/home';
import { DailyReportPage } from './components/daily-reports/daily-report';
import { CustomJobsPage } from './components/custom-jobs/custom-jobs';
import { TickerDetailPage } from './components/ticker/ticker';

export default function App() {
  return (
    <DataProvider>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/daily-report" element={<DailyReportPage />} />
          <Route path="/custom-jobs" element={<CustomJobsPage />} />
          <Route path="/ticker/:ticker" element={<TickerDetailPage />} />
        </Routes>
      </Router>
    </DataProvider>
  );
}