import { createPortal } from "react-dom";
import './ExitPopup.css';

const ExitPopup = ({ onConfirm, onCancel }) => {
    return createPortal(
        <div className="popup-overlay">
            <div className="popup">
                <p>Do you want to exit the slide preview?</p>

                <p>Perform "like" to exit</p>

                <p>Perform "stop" to close this window</p>
            </div>
        </div>,
        document.body
    );
};

export default ExitPopup;

