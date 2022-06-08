import { useState } from "react";
import { FileUploader } from "react-drag-drop-files";

const fileTypes = ["WAV", "MP3"];

function FileLoaderPage(props) {
    const { onLoad } = props;

    return (
        <div className="root">
            <p className="message">Upload a song you want to analyse!</p>

            <FileUploader
                handleChange={onLoad}
                name="file"
                types={fileTypes}
                classes="loader"
                label="Select or drop a file right here"
            />
        </div>
    );
}

export default FileLoaderPage;
