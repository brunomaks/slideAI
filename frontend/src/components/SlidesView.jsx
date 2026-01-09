/*
 Contributors:
- Yaroslav
- Mykhailo

*/

import { useEffect, useRef, useState } from 'react';
import Reveal from 'reveal.js';
import * as pdfjsLib from 'pdfjs-dist';
import 'reveal.js/dist/reveal.css';
import './SlidesView.css';
import ExitDialog from './ExitDialog.jsx';

import { useWebSocket } from '../contexts/WebSocketContext.jsx';

import workerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;

// Gesture mapping configuration
const GESTURE_CONFIG = {
    NAVIGATE_NEXT: {
        gesture: "two_up_inverted",
        direction: "Left"
    },
    NAVIGATE_PREV: {
        gesture: "two_up_inverted",
        direction: "Right"
    },
    OPEN_EXIT_POPUP: {
        gesture: "stop"
    },
    CONFIRM_EXIT: {
        gesture: "like"
    }
};

export default function SlidesView() {
    const [fileURL, setFileURL] = useState(null);
    const fileInputRef = useRef(null);
    const slidesRef = useRef(null);
    const deckRef = useRef(null);
    const revealRootRef = useRef(null);
    const [showDialog, setShowDialog] = useState(false);
    const lastNavigationRef = useRef(0)
    const LOCK_DURATION = 1750
    const CONFIDENCE_THRESHOLD = 0.9

    const { lastMessage: prediction } = useWebSocket();

    const [uiVisible, setUiVisible] = useState(true);

    const openPicker = () => fileInputRef.current?.click();

    const onFileChange = (e) => {
        const file = e.target.files?.[0];
        if (file) {
            const url = URL.createObjectURL(file);
            setFileURL(url);
            setUiVisible(false); // hide UI when file selected
        }
    };

    const exitPreview = () => {
        setFileURL(null);
        setUiVisible(true);
        slidesRef.current.innerHTML = "";
    }

    const matchesGesture = (prediction, config) => {
        if (prediction.predicted_class !== config.gesture) return false;
        if (config.direction && prediction.direction !== config.direction) return false;
        return true;
    };

    useEffect(() => {
        if (!fileURL) return;

        let deck;

        const load = async () => {
            const pdf = await pdfjsLib.getDocument(fileURL).promise;
            const slides = slidesRef.current;
            slides.innerHTML = "";

            for (let i = 1; i <= pdf.numPages; i++) {
                const page = await pdf.getPage(i);
                const viewport = page.getViewport({ scale: 2.5 });

                const canvas = document.createElement("canvas");
                const ctx = canvas.getContext("2d");
                canvas.width = viewport.width;
                canvas.height = viewport.height;

                await page.render({ canvasContext: ctx, viewport }).promise;

                const section = document.createElement("section");
                section.appendChild(canvas);
                slides.appendChild(section);
            }

            deck = new Reveal(revealRootRef.current, {
                controls: false,
                slideNumber: true,
                progress: true,
                hash: false,
                center: false,
                width: '100%',
                height: '100%',
                margin: 0,
                minScale: 1,
                maxScale: 1,
                backgroundTransition: 'none',
                transition: 'fade'
            });

            await deck.initialize();

            deckRef.current = deck

            deck.layout();
        };

        load();

        return () => {
            if (deck) deck.destroy();
        };
    }, [fileURL]);

    useEffect(() => {
        const now = Date.now()
        if (now - lastNavigationRef.current < LOCK_DURATION) return

        lastNavigationRef.current = now

        if (prediction?.confidence < CONFIDENCE_THRESHOLD) {
            console.log("Prediction discarded due to low confidence")
            return
        }

        if (showDialog) {
            if (matchesGesture(prediction, GESTURE_CONFIG.CONFIRM_EXIT)) {
                setShowDialog(false)
                exitPreview()
            } else if(matchesGesture(prediction, GESTURE_CONFIG.OPEN_EXIT_POPUP)) {
                setShowDialog(false)
            }
            return; // critical, otherwise the popup will open up again
        }

        if (!deckRef.current || !prediction) return
        if (!deckRef.current.isReady()) return


        if (matchesGesture(prediction, GESTURE_CONFIG.NAVIGATE_NEXT)) {
            deckRef.current.next()
        } else if (matchesGesture(prediction, GESTURE_CONFIG.NAVIGATE_PREV)) {
            deckRef.current.prev()
        } else if (matchesGesture(prediction, GESTURE_CONFIG.OPEN_EXIT_POPUP)) {
            setShowDialog(true);
        }

    }, [prediction])

    return (
        <div className="slides-view-wrapper">

            {uiVisible && (
                <div className="page-overlay">
                    <div className="page-text">
                        <h1 className="title">Upload your slides</h1>
                        <h2 className="sub-title">
                            Go to your presentation maker software and export your slides as PDF.
                            Upload your slides by pressing the button below and you are ready to go!
                        </h2>
                    </div>

                    <input
                        type="file"
                        accept="application/pdf"
                        ref={fileInputRef}
                        onChange={onFileChange}
                        hidden
                    />

                    <button onClick={openPicker}>Load PDF as Slides</button>

                    <h2 className="hint-title">hint: press F to enter fullscreen mode</h2>
                </div>
            )}

            {fileURL && (
                <>
                    {!uiVisible && (
                        <button className="exit-button" onClick={exitPreview}>
                            Exit Preview
                        </button>

                    )}
                    {!uiVisible && showDialog && (
                        <ExitDialog/>
                    )}
                    <div className="reveal" ref={revealRootRef}>
                        <div className="slides" ref={slidesRef}></div>
                    </div>
                </>
            )}

        </div>
    );


}
