<!-- Update your JavaScript code -->
<script>
    $(document).ready(function () {
        $('#essayForm').submit(function (event) {
            event.preventDefault();  // Prevent the default form submission

            let essayText = $('#essay_text').val();
            let prompt = $('#prompt').val();

            $.ajax({
                type: 'POST',
                url: '/predict',
                data: { 'essay_text': essayText, 'prompt': prompt },
                success: function (data) {
                    // Update the result element with the prediction
                    $('#result').text('Predicted Score: ' + data.prediction[0]);
                }
            });
        });
    });
</script>
