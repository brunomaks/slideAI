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

        const load = async () => {
            const pdf = await pdfjsLib.getDocument(fileURL).promise;
            console.log(`PDF loaded with ${pdf.numPages} pages`);
            const slides = slidesRef.current;
            slides.innerHTML = '';

            for (let i = 1; i <= pdf.numPages; i++) {
                const page = await pdf.getPage(i);
                const viewport = page.getViewport({ scale: 1.5 });

                const canvas = document.createElement('canvas');
                canvas.width = viewport.width;
                canvas.height = viewport.height;
                const ctx = canvas.getContext('2d');

                await page.render({ canvasContext: ctx, viewport }).promise;

                const section = document.createElement('section');
                section.appendChild(canvas);
                console.log(`Page ${i} rendered`);
                slides.appendChild(section);
            }

            Reveal.initialize({
                width: 960,
                height: 700,
                embedded: false,
                hash: false,
                controls: true,
                progress: true,
            });
        };

        load();
    }, [fileURL]);

    return (
        <div className='w-full h-full p-4 flex flex-col'>
            <h1 className='text-1xl font-bold'>Upload your slides</h1>
            <input
                type='file'
                accept='application/pdf'
                ref={fileInputRef}
                onChange={onFileChange}
                hidden={true}
            />

            <button onClick={openPicker}>
                Load PDF as Slides
            </button>

            <div>
                <div className='slides' ref={slidesRef}></div>
            </div>
        </div>
    );
}
