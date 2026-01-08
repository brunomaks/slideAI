/*
 Contributors:
- Ahmet
- 

*/

import './LandingPage.css';

export default function Landing({ onEnter }) {
  return (
    <div className="main-page">
      <div className="main-content">
        <h1 className="product-name">SlideAI</h1>
        <p className="pitch">
          Contol your presenetation with simple intuive gestures
        </p>
        <p className="description">
          SlideAI tracks your hands in real time, recognizes gestures and performs control commands. 
          No need to touch anything.
        </p>
        <button className="start-button" onClick={onEnter}>
          Enter SlideAI
        </button>
      </div>
    </div>
  );
}
