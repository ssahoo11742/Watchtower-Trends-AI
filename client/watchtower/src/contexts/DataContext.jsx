// src/contexts/DataContext.jsx
import { createContext, useContext, useState } from 'react';
import Papa from 'papaparse';

const DataContext = createContext();

export const useData = () => {
  const context = useContext(DataContext);
  if (!context) {
    throw new Error('useData must be used within DataProvider');
  }
  return context;
};

export const DataProvider = ({ children }) => {
  const [data, setData] = useState([]);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        setData(results.data);
      },
      error: (error) => {
        console.error('Error parsing CSV:', error);
      }
    });
  };

  return (
    <DataContext.Provider value={{ data, handleFileUpload }}>
      {children}
    </DataContext.Provider>
  );
};