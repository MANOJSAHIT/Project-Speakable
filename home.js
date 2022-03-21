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
var globalStream;
$("#startCameraButton").on("click", function(){
    if(!document.getElementById("liveVideo"))
    {
        var video = document.createElement('video');
        video.setAttribute('playsinline', '');
        video.setAttribute('autoplay', '');
        video.setAttribute('muted', '');
        video.setAttribute('id', 'liveVideo');
        video.style.width = '100%';
        video.style.height = '100%';
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
    }
    else
    {
        globalStream.getTracks().forEach(function(track){
            if (track.readyState == 'live'){
                track.stop();
            }
        });
        document.getElementById("liveVideo").parentNode.removeChild(document.getElementById("liveVideo"));
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
        document.getElementById("liveVideo").parentNode.removeChild(document.getElementById("liveVideo"));
        $(".camera-container").css("display", "block");
        $("#startCameraButton").text("Start Camera and Recognise the Sign Language");
        $("#startCameraButton").removeClass('active');
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
    }
    else
    {
        $("#voiceOptions").html("<option data-lang='hi-IN' data-name='Google (hi-IN)'>Google हिन्दी</option>");
    }
});
/*function OnLoad()
{
    var control = new google.elements.transliteration.TransliterationControl({sourceLanguage: google.elements.transliteration.LanguageCode.ENGLISH, destinationLanguage: [google.elements.transliteration.LanguageCode.HINDI], shortcutKey: 'ctrl+g', transliterationEnabled: true});
    control.makeTransliteratable(["speakText"]);
}
google.setOnLoadCallback(OnLoad);*/
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
            console.log(voices[i]);
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