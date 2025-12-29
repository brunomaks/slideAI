import { useState } from 'react'
import Home from './pages/Home'
import Landing from './pages/LandingPage'
import Topbar from './components/TopBar'
import { WebSocketProvider } from './contexts/WebSocketContext'

function App() {
  const [showLanding, setShowLanding] = useState(false);

  const handleEnter = () => {
    setShowLanding(false);
  };

  const handleReset = () => {
    setShowLanding(true);
  };

  return (
    <WebSocketProvider>
      <Topbar onBrandClick={handleReset} />
      {showLanding && <Landing onEnter={handleEnter} />}
      {!showLanding && (
        <Home />
      )}
    </WebSocketProvider>
  )
}

export default App
