/*
 Contributors:
- Ahmet
- Yaroslav

*/

import StreamsView from "../components/StreamsView";
import SlidesView from "../components/SlidesView";
import './Home.css';

export default function Home() {
    return (
        <div className="home-page">
            <div className="streams-overlay">
                <StreamsView />
            </div>

            <SlidesView />
        </div>
    );
}
