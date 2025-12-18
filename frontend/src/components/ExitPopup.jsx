// We don't want to render the popup inside Reveal or PDF containers. 
// Portal allows to render it into document.body
import { createPortal } from "react-dom"; // we 
import './ExitPopup.css';

const ExitPopup = () => {
    return createPortal(
        <div className="popup-overlay">
            <div className="popup">
                <p className="popup-title">Do you want to exit the slide preview?</p>

                <p className="popup-instruction">Perform <strong>"like"</strong> to exit</p>

                <p className="popup-instruction">Perform <strong>"stop"</strong> to close this window</p>
            </div>
        </div>,
        document.body
    );
};

export default ExitPopup;