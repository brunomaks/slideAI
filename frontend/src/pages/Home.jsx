import StreamsView from "../components/StreamsView";
import SlidesView from "../components/SlidesView";

export default function Home() {
    return (
        <div className="home-page">
            <StreamsView />
            <SlidesView />
        </div>
    );
}
