[...document.querySelectorAll('.menu-button')].forEach(function(item){
    item.addEventListener('click', function()
    {
        document.querySelector('.app-left').classList.add('show');
    });
});
[...document.querySelectorAll('.close-menu')].forEach(function(item){
    item.addEventListener('click', function()
    {
        document.querySelector('.app-left').classList.remove('show');
    });
});
$("#recogniseButton").click(function()
{
    $("#dictionaryAppMain").css("display", "none");
    $("#dictionary1AppMain").css("display", "none");
    $("#teamAppMain").css("display", "none");
    $("#recogniseAppMain").css("display", "block");
    $(this).addClass('active').siblings().removeClass('active');
    if(document.querySelector('.app-left').classList.contains('show'))
    {
        document.querySelector('.app-left').classList.remove('show');
    }
});
$("#dictionaryButton").click(function()
{
    $("#recogniseAppMain").css("display", "none");
    $("#teamAppMain").css("display", "none");
    $("#dictionary1AppMain").css("display", "none");
    $("#dictionaryAppMain").css("display", "block");
    $(this).addClass('active').siblings().removeClass('active');
    if(document.querySelector('.app-left').classList.contains('show'))
    {
        document.querySelector('.app-left').classList.remove('show');
    }
});
$("#dictionary1Button").click(function()
{
    $("#recogniseAppMain").css("display", "none");
    $("#teamAppMain").css("display", "none");
    $("#dictionaryAppMain").css("display", "none");
    $("#dictionary1AppMain").css("display", "block");
    $(this).addClass('active').siblings().removeClass('active');
    if(document.querySelector('.app-left').classList.contains('show'))
    {
        document.querySelector('.app-left').classList.remove('show');
    }
});
$("#teamButton").click(function()
{
    $("#recogniseAppMain").css("display", "none");
    $("#dictionaryAppMain").css("display", "none");
    $("#dictionary1AppMain").css("display", "none");
    $("#teamAppMain").css("display", "block");
    $(this).addClass('active').siblings().removeClass('active');
    if(document.querySelector('.app-left').classList.contains('show'))
    {
        document.querySelector('.app-left').classList.remove('show');
    }
});
var bringPredictions;
var globalStream;
var video;
var canvas = document.querySelector("#canvas");
var adaptiveCanvas = document.querySelector("#adaptiveCanvas");
var interval=400;
var dataset = 1;
function changeLanguage()
{
    var checkbox = document.getElementsByClassName("checkbox")[0];
    if(checkbox.checked==true)
    {
        dataset = 2;
        interval = 2000;
    }
    else
    {
        dataset = 1;
        interval = 400;
    }
}
$("#startCameraButton").on("click", function(){
    if(!document.getElementById("liveVideo"))
    {
        video = document.createElement('video');
        video.setAttribute('playsinline', '');
        video.setAttribute('autoplay', '');
        video.setAttribute('muted', '');
        video.setAttribute('id', 'liveVideo');
        video.style.display="none";
        video.style.width = '640px';
        video.style.height = '480px';
        var facingMode = "user";
        var constraints = {audio: false,video: {facingMode: facingMode}};
        navigator.mediaDevices.getUserMedia(constraints).then(function success(stream){
            video.srcObject = stream;
            globalStream = stream;
        });
        $(".camera-container").css("display", "none");
        var mainContentData = document.getElementsByClassName("main-content-data")[0];
        mainContentData.appendChild(video);
        $("#startCameraButton").text("Stop recognition and reset the Prediction Panel");
        $("#startCameraButton").addClass('active');
        $("#adaptiveCanvas").css("display", "block");
        bringPredictions = setInterval(function(){
            canvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
            var image_data_url = canvas.toDataURL('image/jpeg');
            $.ajax({
                type: "POST",
                contentType: "application/json; charset=utf-8",
                url: "/recognise",
                data: JSON.stringify({imageData: image_data_url, dataset:dataset}),
                success: function(data){
                    $("#predictionPanelCharacter").text(data["character"]);
                    $("#predictionPanelWord").text(data["word"]);
                    $("#predictionPanelSentence").text(data["sentence"]);
                    adaptiveCanvas.src=data["adaptiveImg"];
                },
                dataType: "json"
            });
        }, interval);
    }
    else
    {
        globalStream.getTracks().forEach(function(track){
            if (track.readyState == 'live'){
                track.stop();
            }
        });
        $("#adaptiveCanvas").css("display", "none");
        document.getElementById("liveVideo").parentNode.removeChild(document.getElementById("liveVideo"));
        clearInterval(bringPredictions);
        $(".camera-container").css("display", "block");
        $("#startCameraButton").removeClass('active');
        $("#startCameraButton").text("Start Camera and Recognise the Sign Language");
        $("#predictionPanelCharacter").text("-");
        $("#predictionPanelWord").text("-");
        $("#predictionPanelSentence").text("-");
    }
});
$("#moveSentenceButton").on("click", function(){
    var sentenceToMove = $("#predictionPanelSentence").text();
    if(sentenceToMove!="-")
    {
        globalStream.getTracks().forEach(function(track){
            if (track.readyState == 'live'){
                track.stop();
            }
        });
        $("#adaptiveCanvas").css("display", "none");
        document.getElementById("liveVideo").parentNode.removeChild(document.getElementById("liveVideo"));
        clearInterval(bringPredictions);
        $(".camera-container").css("display", "block");
        $("#startCameraButton").removeClass('active');
        $("#startCameraButton").text("Start Camera and Recognise the Sign Language");
        $("#predictionPanelCharacter").text("-");
        $("#predictionPanelWord").text("-");
        $("#predictionPanelSentence").text("-");
        $("#speakText").val(sentenceToMove);
    }
});
if (!window.speechSynthesis)
{
    $("#warning").css("display", "block");
    $("#speak").css("display", "none");
}
$("#languageOptions").change(function(){
    var language = $("#languageOptions :selected").val();
    if(language=='en')
    {
        $("#voiceOptions").html("<option data-lang='en-IN' data-name='Microsoft Heera - English (India)'>Microsoft Heera - English (India)</option><option data-lang='en-IN' data-name='Microsoft Ravi - English (India)'>Microsoft Ravi - English (India)</option>");
        $("#speakText").val(window.speakText);
    }
    else
    {
        $("#voiceOptions").html("<option data-lang='hi-IN' data-name='Google हिन्दी'>Google हिन्दी</option>");
        window.speakText = $("#speakText").val();
        const settings = {
            "async": true,
            "crossDomain": true,
            "url": "https://google-translate1.p.rapidapi.com/language/translate/v2",
            "method": "POST",
            "headers": {
                "content-type": "application/x-www-form-urlencoded",
                "X-RapidAPI-Host": "<RAPID-API-HOST>",
                "X-RapidAPI-Key": "<RAPID-API-KEY>"
            },
            "data": {
                "q": $("#speakText").val(),
                "target": "hi",
                "source": "en"
            }
        };
        $.ajax(settings).done(function(response){
            $("#speakText").val(response['data']['translations'][0]['translatedText']);
        });
    }
});
$("#speak").on("submit",function(event){
    event.preventDefault();
    var voiceSelect = document.getElementById("voiceOptions");
    var utterThis=new SpeechSynthesisUtterance($("#speakText").val());
    var selectedOption=voiceSelect.selectedOptions[0].getAttribute('data-name');
    var voices = window.speechSynthesis.getVoices();
    for(i=0;i<voices.length;i++)
    {
        if(voices[i].name===selectedOption)
        {
            utterThis.voice=voices[i];
        }
    }
    window.speechSynthesis.speak(utterThis);
});
function getScrollHeight(elm)
{
    var savedValue = elm.value
    elm.value = ''
    elm._baseScrollHeight = elm.scrollHeight
    elm.value = savedValue
}
document.addEventListener('input', function({target:elm}){
    if(!elm.classList.contains('autoExpand')||!elm.nodeName=='TEXTAREA')
    {
        return;
    }
    var minRows = elm.getAttribute('data-min-rows')|0, rows;
    !elm._baseScrollHeight && getScrollHeight(elm);
    elm.rows = minRows;
    rows = Math.ceil((elm.scrollHeight - elm._baseScrollHeight) / 16);
    elm.rows = minRows + rows;
});
