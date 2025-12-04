import './LandingPage.css';

export default function Landing({ onEnter }) {
  return (
    <div className="main-page">
      <div className="main-content">
        <h1 className="product-name">Gesturify</h1>
        <p className="pitch">
          Your hand, your Hadouken. Master the stroke that commands the storm.
        </p>
        <p className="description">
          Gesturify tracks your hands in real time, recognizes fighting gestures 
          like punches and energy blasts, and maps them to actions so you can play 
          a lightweight fighting game using only your body.
        </p>
        <button className="start-button" onClick={onEnter}>
          Enter Gesturify
        </button>
      </div>
    </div>
  );
}
