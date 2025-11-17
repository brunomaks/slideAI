import { useState } from 'react'
import Home from './pages/Home'
import Landing from './pages/LandingPage'
import Topbar from './components/TopBar'

function App() {
  const [showLanding, setShowLanding] = useState(true);

  const handleEnter = () => {
    setShowLanding(false);
  };

  const handleReset = () => {
    setShowLanding(true);
  };

  return (
    <>
      <Topbar onBrandClick={handleReset} />
      {showLanding && <Landing onEnter={handleEnter} />}
      {!showLanding && (
        <Home />
      )}
    </>
  )
}

export default App
