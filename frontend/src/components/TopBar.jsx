/*
 Contributors:
- Ahmet
- Mahmoud

*/

import '../pages/LandingPage.css';

export default function Topbar({ onBrandClick }) {
  return (
    <nav className="topbar">
      <div className="topbar-content">
        <div className="topbar-brand" onClick={onBrandClick}>SlideAI</div>
        <a
          href="https://git.chalmers.se/courses/dit826/2025/team4#data-intensive-ai-applications"
          target="_blank"
          rel="noopener noreferrer"
          className="topbar-button"
        >
          Documentation
        </a>
      </div>
    </nav>
  );
}
