import { useState } from 'react'
import Home from './pages/Home'
import Landing from './pages/LandingPage'
import Topbar from './components/TopBar'
import { WebRTCProvider } from './contexts/WebRTCContext'

function App() {
  const [showLanding, setShowLanding] = useState(false);

  const handleEnter = () => {
    setShowLanding(false);
  };

  const handleReset = () => {
    setShowLanding(true);
  };

  return (
    <WebRTCProvider>
      <Topbar onBrandClick={handleReset} />
      {showLanding && <Landing onEnter={handleEnter} />}
      {!showLanding && (
        <Home />
      )}
    </WebRTCProvider>
  )
}

export default App
