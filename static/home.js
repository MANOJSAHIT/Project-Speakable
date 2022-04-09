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
function changeLanguage()
{
    var checkbox = document.getElementsByClassName("checkbox")[0];
    if(checkbox.checked==true)
    {
        $.getJSON('/changeToISL');
    }
    else
    {
        $.getJSON('/changeToASL');
    }
}
var bringPredictions;
$("#startCameraButton").on("click", function(){
    if($("#liveVideo").css("display")=='none')
    {
        $("#liveVideo").css("display", "block");
        $(".camera-container").css("display", "none");
        $("#startCameraButton").text("Stop recognition and reset the Prediction Panel");
        $("#startCameraButton").addClass('active');
        $.getJSON('/startRecognising');
        bringPredictions = setInterval(function(){
            $.getJSON('/getData', function(data)
            {
                $("#predictionPanelCharacter").text(data["character"]);
                $("#predictionPanelWord").text(data["word"]);
                $("#predictionPanelSentence").text(data["sentence"]);
            });
        }, 1000);
    }
    else
    {
        $("#liveVideo").css("display", "none");
        clearInterval(bringPredictions);
        $.getJSON('/stopRecognising');
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
        $("#liveVideo").css("display", "none");
        clearInterval(bringPredictions);
        $.getJSON('/stopRecognising');
        $(".camera-container").css("display", "block");
        $("#startCameraButton").text("Start Camera and Recognise the Sign Language");
        $("#startCameraButton").removeClass('active');
        $("#predictionPanelCharacter").text("-");
        $("#predictionPanelWord").text("-");
        $("#predictionPanelSentence").text("-");
        $("#speakText").val(sentenceToMove);
    }
});
window.addEventListener('beforeunload', function(e){
    e.preventDefault();
    $.getJSON('/stopRecognising');
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
        /*const settings = {
            "async": true,
            "crossDomain": true,
            "url": "https://google-translate1.p.rapidapi.com/language/translate/v2",
            "method": "POST",
            "headers": {
                "content-type": "application/x-www-form-urlencoded",
                "X-RapidAPI-Host": "google-translate1.p.rapidapi.com",
                "X-RapidAPI-Key": "256de473femsh9695bb1de09d790p1700aejsn90930725d0ea"
            },
            "data": {
                "q": $("#speakText").val(),
                "target": "hi",
                "source": "en"
            }
        };
        $.ajax(settings).done(function(response){
            $("#speakText").val(response['data']['translations'][0]['translatedText']);
        });*/
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