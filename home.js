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
    $("#dictionaryAppMain").css("display", "block");
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
        $("#startCameraButton").text("Camera is running...");
    }
});
$("#moveSentenceButton").on("click", function(){
    var sentenceToMove = $("#controlPanelSentence").text();
    if(sentenceToMove!="-")
    {
        globalStream.getTracks().forEach(function(track){
            if (track.readyState == 'live'){
                track.stop();
            }
        });
        document.getElementById("liveVideo").parentNode.removeChild(document.getElementById("liveVideo"));
        $("#startCameraButton").text("Start Camera and Recognise ASL");
        $(".camera-container").css("display", "block");
        $("#controlPanelCharacter").text("-");
        $("#controlPanelWord").text("-");
        $("#controlPanelSentence").text("-");
        $("#speakText").val(sentenceToMove);
    }
});
if (!window.speechSynthesis)
{
    $("#warning").css("display", "block");
    $("#speak").css("display", "none");
}
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