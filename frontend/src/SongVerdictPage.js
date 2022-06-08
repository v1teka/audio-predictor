function SongVerdictPage(props) {
    const { songName, isPopular, confidence, onReset } = props;

    const verdictString = isPopular ? (
        <>
            Wow! <b>{songName}</b> is likely to become popular.
        </>
    ) : (
        <>
            <b>{songName}</b> is less likely to become popular...
        </>
    );

    const emoji = isPopular ? "🤟" : "🤔";
    const displayedConfidence = Math.round(confidence * 10000) / 100;

    return (
        <div className={isPopular ? "root good" : "root bad"}>
            <p className="emoji">{emoji}</p>
            <p className="message">{verdictString}</p>
            <p className="confidence">{displayedConfidence}%</p>
            <button className="button" onClick={onReset}>
                Try another one
            </button>
            <p className="hint">
                * probability of being popular according to our analysis
            </p>
        </div>
    );
}

export default SongVerdictPage;
