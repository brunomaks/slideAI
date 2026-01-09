import { useState } from 'react'
import Home from './pages/Home'
import Landing from './pages/LandingPage'
import Topbar from './components/TopBar'
import { WebSocketProvider } from './contexts/WebSocketContext'

const WEBSOCKET_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8001/ws/landmarks/'
const MAX_RECONNECT_ATTEMPS = 5
const RECONNECT_DELAY = 3000
const AUTO_RECONNECT = true

function App() {
  const [showLanding, setShowLanding] = useState(true);

  const handleEnter = () => {
    setShowLanding(false);
  };

  const handleReset = () => {
    setShowLanding(true);
  };

  return (
    <WebSocketProvider 
      url={WEBSOCKET_URL}
      options={{
        maxReconnectAttempts: MAX_RECONNECT_ATTEMPS,
        reconnectDelay: RECONNECT_DELAY,
        autoReconnect: AUTO_RECONNECT
      }}
    >
      <Topbar onBrandClick={handleReset} />
      {showLanding && <Landing onEnter={handleEnter} />}
      {!showLanding && (
        <Home />
      )}
    </WebSocketProvider>
  )
}

export default App
