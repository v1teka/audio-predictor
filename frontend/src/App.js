import { useState } from "react";

import axios from "axios";

import FileLoaderPage from "./FileLoaderPage";
import SongVerdictPage from "./SongVerdictPage";

import "./App.css";

const apiURL = "http://localhost:5001/"; // local backend URL
const axiosClient = axios.create({ baseURL: apiURL, withCredentials: false });

const isPopularLike = async (songBlobData) => {
    const formData = new FormData();

    formData.append("audio_file", songBlobData, songBlobData.name);

    const {
        data: { probability, verdict },
    } = await axiosClient.post("/analyze", formData, {
        headers: {
            "content-type": "multipart/form-data",
        },
    });

    console.log({ verdict, probability });

    return { verdict, confidence: probability };
};

function App() {
    const [loading, setLoading] = useState(false);
    const [songFile, setSongFile] = useState();
    const [isSongPopularLike, setIsSongPopularLike] = useState(false);
    const [predictionConfidence, setPredictionConfidence] = useState(0);

    const handleFileData = async (songBlobData) => {
        if (!songBlobData) return;

        setLoading(true);
        setSongFile(songBlobData);

        const { verdict, confidence } = await isPopularLike(songBlobData);

        console.log({ verdict });

        setIsSongPopularLike(verdict);
        setPredictionConfidence(confidence);
        setLoading(false);
    };

    const handleResetClick = () => {
        setSongFile(null);
    };

    if (!songFile?.name) return <FileLoaderPage onLoad={handleFileData} />;

    if (loading)
        return (
            <div className="root">
                <p className="emoji">🎧</p>

                <p className="message">
                    Analysing <b>{songFile.name}</b>...
                </p>
            </div>
        );

    return (
        <SongVerdictPage
            songName={songFile.name}
            isPopular={isSongPopularLike}
            confidence={predictionConfidence}
            onReset={handleResetClick}
        />
    );
}

export default App;
