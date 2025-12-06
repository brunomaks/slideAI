import { useEffect, useRef, useState } from 'react';
import Reveal from 'reveal.js';
import * as pdfjsLib from 'pdfjs-dist';
import 'reveal.js/dist/reveal.css';
import './SlidesView.css';

import workerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;

export default function SlidesView() {
    const [fileURL, setFileURL] = useState(null);
    const fileInputRef = useRef(null);
    const slidesRef = useRef(null);

    const openPicker = () => fileInputRef.current?.click();
    const onFileChange = (e) => {
        const file = e.target.files?.[0];
        if (file) {
            const url = URL.createObjectURL(file);
            setFileURL(url);
        }
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
                const viewport = page.getViewport({ scale: 1.5 });

                const canvas = document.createElement("canvas");
                const ctx = canvas.getContext("2d");
                canvas.width = viewport.width;
                canvas.height = viewport.height;

                await page.render({ canvasContext: ctx, viewport }).promise;

                const section = document.createElement("section");
                section.appendChild(canvas);
                slides.appendChild(section);
            }

            deck = new Reveal({
                controls: true,
                progress: true,
                hash: false,
                width: 960,
                height: 700,
            });

            await deck.initialize();
            deck.layout();
        };

        load();

        return () => {
            if (deck) deck.destroy();
        };
    }, [fileURL]);

    return (
        <div className='w-full h-full p-4 flex flex-col'>
            <div className='page'>
                <div className='page-text'>
                    <h1 className='title'>Upload your slides</h1>
                    <h2 className='sub-title'>Go to your presentation maker software and export your slides as PDF.
                        Upload your slides by pressing the button below and you are ready to go!</h2>
                </div>
                <input
                    type='file'
                    accept='application/pdf'
                    ref={fileInputRef}
                    onChange={onFileChange}
                    hidden={true}
                />
                <button onClick={openPicker} className='upload-slides'>Load PDF as Slides</button>
                <h2 className='hint-title'>hint: press F to enter fullscreen mode</h2>
            </div>
            <div className="reveal">
                <div className="slides" ref={slidesRef}></div>
            </div>
        </div>
    );
}
