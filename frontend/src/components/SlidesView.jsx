import { useEffect, useRef, useState } from 'react';
import Reveal from 'reveal.js';
import * as pdfjsLib from 'pdfjs-dist';
import 'reveal.js/dist/reveal.css';
import './SlidesView.css';

import { useWebRTC } from '../contexts/WebRTCContext.jsx';

import workerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;

export default function SlidesView() {
    const [fileURL, setFileURL] = useState(null);
    const fileInputRef = useRef(null);
    const slidesRef = useRef(null);
    const deckRef = useRef(null);
    const revealRootRef = useRef(null);

    const { prediction } = useWebRTC();

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
                progress: true,
                hash: false,
                center: false,
                width: '100%',
                height: '100%',
                margin: 0,
                minScale: 1,
                maxScale: 1,
                backgroundTransition: 'none',
                transition: 'none'
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
        console.log("Effect triggered, deckRef:", !!deckRef.current, "prediction:", prediction)
        if (!deckRef.current || !prediction) {
            console.log("Early return: missing deck or prediction")
            return
        }
        if (!deckRef.current.isReady()) {
            console.log("Early return: deck not ready")
            return
        }

        console.log("Slide change:", prediction.predicted_class)

        console.log("Reveal indices:", deckRef.current.getIndices())

        switch (prediction.predicted_class) {
            case "right":
                deckRef.current.next()
                break
            case "left":
                deckRef.current.prev()
                break
            default:
                break
        }
    }, [prediction?.predicted_class])

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
                    <div className="reveal" ref={revealRootRef}>
                        <div className="slides" ref={slidesRef}></div>
                    </div>
                </>
            )}
        </div>
    );


}
